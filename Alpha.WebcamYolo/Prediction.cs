using System.Drawing;

namespace Alpha.WebcamYolo
{
    public class Prediction
    {
        public string? Label { get; init; }
        public RectangleF Rectangle { get; init; }
        public float Score { get; init; }
    }
}