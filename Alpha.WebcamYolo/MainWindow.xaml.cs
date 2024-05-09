using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace Alpha.WebcamYolo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private DispatcherTimer _timer;
        private InferenceSession _session;

        static string[] classesNames = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };



        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
            Closed += MainWindow_Closed;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {

            SessionOptions options = new SessionOptions();
            // Check if GPU is available and add CUDA as the execution provider

            //  using var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0); check Cuda Avaibility first




            _session = new InferenceSession("yolov8m.onnx"); // YOLOv8 ONNX model filename
            _capture = new VideoCapture();
         
            _capture.ImageGrabbed += ProcessFrame;
            _capture.Start();
           
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            var frame = new Mat();


            _capture.Retrieve(frame, 0);
            CvInvoke.Resize(frame, frame, new System.Drawing.Size(640, 640));
            var start = DateTime.Now;

            var processedFrame = PerformInference(frame);
            var end = DateTime.Now - start;
            var bitmap = BitmapSourceConvert.ToBitmapSource(processedFrame);

            bitmap.Freeze(); // Freeze the bitmap for UI thread usage.
            Dispatcher.Invoke(() => VideoImage.Source = bitmap);
        }
        private Mat PerformInference(Mat frame)
        {
            var inputTensor = PreprocessImage(frame, 640, 640);
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.First().Key, inputTensor)
        };

            using var results = _session.Run(inputs);
           return Postprocess(frame, results);

           
        }

        private Mat Postprocess(Mat img, IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            try
            {
                var output = results.First().AsEnumerable<float>().ToArray();
                var tensor = new DenseTensor<float>(output, new[] { 1, 84, 8400 });

                var shape = tensor.Dimensions;

                var result = ParseOutput(tensor, img);
                var finalResult = Suppress(result);
                Console.Write(result);
                var imageResult = DrawPredictions(finalResult, img);
                return imageResult;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return null;

            }
        }

        private Mat DrawPredictions(Prediction[] finalResult, Mat img)
        {
            Mat resultMat = img.Clone();

            // Iterate over predictions and draw on the image
            foreach (var prediction in finalResult)
            {
                // Draw rectangle
                System.Drawing.Rectangle rect = new System.Drawing.Rectangle((int)prediction.Rectangle.X, (int)prediction.Rectangle.Y, (int)prediction.Rectangle.Width, (int)prediction.Rectangle.Height);
                CvInvoke.Rectangle(resultMat, rect, new MCvScalar(0, 0, 255), 2); // Red color, thickness 2

                // Draw label
                System.Drawing.Point labelPosition = new System.Drawing.Point((int)(prediction.Rectangle.X), (int)(prediction.Rectangle.Y)); // Adjust label position as needed
                CvInvoke.PutText(resultMat, prediction.Label, labelPosition, FontFace.HersheyComplex, 0.5, new MCvScalar(0, 0, 255), 1); // Red color, thickness 1
               
            }

            return resultMat;
        }
    

        protected static Prediction[] Suppress(List<Prediction> predictions)
        {
            var Overlap = 0.5;
            var result = new List<Prediction>(predictions);

            foreach (var item in predictions) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result.ToArray();
        }


        protected static List<Prediction> ParseOutput(DenseTensor<float> output, Mat image)
        {
            var ModelInputWidth = 640;
            var ModelInputHeight = 640;
            var result = new ConcurrentBag<Prediction>();
            var Confidence = 0.7;
            var ModelOutputDimensions = 84;
            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (ModelInputWidth / (float)w, ModelInputHeight / (float)h); // x, y gains
            var (xPad, yPad) = ((ModelInputWidth - w * xGain) / 2, (ModelInputHeight - h * yGain) / 2); // left, right pads

            //for each batch
            for (int i = 0; i < output.Dimensions[0]; i++)
            {
                // divide total length by the elements per prediction
                for (int j = 0; j < (int)(output.Length / output.Dimensions[1]); j++)
                {
                    float xMin = ((output[i, 0, j] - output[i, 2, j] / 2) - xPad) / xGain; // unpad bbox tlx to original
                    float yMin = ((output[i, 1, j] - output[i, 3, j] / 2) - yPad) / yGain; // unpad bbox tly to original
                    float xMax = ((output[i, 0, j] + output[i, 2, j] / 2) - xPad) / xGain; // unpad bbox brx to original
                    float yMax = ((output[i, 1, j] + output[i, 3, j] / 2) - yPad) / yGain; // unpad bbox bry to original

                    xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    for (int l = 0; l < ModelOutputDimensions - 4; l++)
                    {
                        var pred = output[i, 4 + l, j];

                        // skip low confidence values
                        if (pred < Confidence) continue;

                        result.Add(new Prediction()
                        {
                            Label = classesNames[l],
                            Score = pred,
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    }
                }
            }

            return result.ToList();
        }

        private static Tensor<float> PreprocessImage(Mat image, int inputWidth, int inputHeight)
        {
            Image<Bgr, Byte> img = image.ToImage<Bgr, Byte>();

            var tensor = new DenseTensor<float>(new[] { 1, 3, inputHeight, inputWidth }); // NCHW format


            // Process the pixels
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    tensor[0, 0, y, x] = img.Data[x, y, 2] / 255.0f; // Normalize and assign Red channel
                    tensor[0, 1, y, x] = img.Data[x, y, 1] / 255.0f; // Normalize and assign Green channel
                    tensor[0, 2, y, x] = img.Data[x, y, 0] / 255.0f; // Normalize and assign Blue channel
                }
            }

            return tensor;
        }

        private void MainWindow_Closed(object sender, EventArgs e)
        {
            _capture.Dispose();
        }
    }

    public static class BitmapSourceConvert
    {
        public static BitmapSource ToBitmapSource(Mat image)
        {
            var pixelData = new byte[image.Width * image.Height * image.ElementSize];
            Marshal.Copy(image.DataPointer, pixelData, 0, pixelData.Length);

            var bitmap = BitmapSource.Create(image.Width, image.Height, 96, 96, System.Windows.Media.PixelFormats.Bgr24, null, pixelData, image.Width * image.ElementSize);
            bitmap.Freeze(); // To prevent caching of images by the host application.
            return bitmap;
        }
    }
}