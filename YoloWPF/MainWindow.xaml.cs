using Emgu.CV;
using Emgu.CV.CvEnum;
using Microsoft.Extensions.Configuration;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Configuration;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
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
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using Image = SixLabors.ImageSharp.Image;

namespace YoloWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Yolo _yolo;
        private Dispatcher _dispatcher;
        private CancellationTokenSource _webcamCancellationTokenSource;
        private CancellationToken _webcamCancellationToken;

        public IConfiguration Configuration { get; private set; }


        public MainWindow()
        {
            var builder = new ConfigurationBuilder()
  .SetBasePath(Directory.GetCurrentDirectory())
  .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);

            Configuration = builder.Build();
            var modelPath = Configuration.GetSection("Model").Value;
            var cudaUsage = bool.Parse(Configuration.GetSection("CudaUsage").Value);


            if (cudaUsage)

                try
                {

                    _yolo = new Yolo(modelPath, cuda: true);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error during Inferecing with cuda, Moving to CPU !");
                    MessageBox.Show(ex.Message);
                    _yolo = new Yolo(modelPath, cuda: false);
                }
            else
                _yolo = new Yolo(modelPath, cuda: false);



            _dispatcher = Dispatcher.CurrentDispatcher;
            _webcamCancellationTokenSource = new CancellationTokenSource();
            _webcamCancellationToken = _webcamCancellationTokenSource.Token;
            //Invoke webcam in a new thread
            Task.Run(() => WebcamAsync(_webcamCancellationToken), _webcamCancellationToken);
            InitializeComponent();
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {

            _webcamCancellationTokenSource = new CancellationTokenSource();
            _webcamCancellationToken = _webcamCancellationTokenSource.Token;
            //Invoke webcam in a new thread
            Task.Run(() => WebcamAsync(_webcamCancellationToken), _webcamCancellationToken);
        }
        private async Task WebcamAsync(CancellationToken cancellationToken)
        {
            try
            {
                using var capture = new VideoCapture(0, VideoCapture.API.DShow);
                capture.Set(property: CapProp.FrameCount, value: 30);
                capture.Set(property: CapProp.FrameWidth, value: 640);
                capture.Set(property: CapProp.FrameHeight, value: 640);

                using MemoryStream? stream = new MemoryStream();

                while (cancellationToken.IsCancellationRequested is false)
                {
                    capture.QueryFrame().ToBitmap().Save(stream, format: ImageFormat.Bmp);
                    stream.Position = 0;
                    using Image<Bgra32> img = await Image.LoadAsync<Bgra32>(stream);
                    List<ObjectDetection?> results = _yolo.RunObjectDetection(img);
                    img.Draw(results);
                    await _dispatcher.Invoke(async () => WebcamImage.Source = await ImageSharpToBitmapAsync(img));

                }


            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
                throw ex;
            }
        }

        private static async Task<ImageSource> ImageSharpToBitmapAsync(Image<Bgra32> image)
        {
            using MemoryStream ms = new MemoryStream();
            await image.SaveAsBmpAsync(ms);
            ms.Position = 0;
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = ms;
            bitmap.EndInit();
            return bitmap;
        }

 
    }
}