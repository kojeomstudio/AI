using System.IO;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;
using MabinogiMacro.Models;
using MabinogiMacro.Native;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;

namespace MabinogiMacro.Services;

public class YoloDetectionService : IDisposable
{
    private InferenceSession? _session;
    private readonly ElementMapping _elementMapping;
    private readonly int _modelInputSize;
    private readonly Dictionary<int, ElementType> _classIdToElementType = new();
    private bool _disposed;

    public bool IsModelLoaded => _session != null;

    public YoloDetectionService(ElementMapping elementMapping, int modelInputSize = 640)
    {
        _elementMapping = elementMapping;
        _modelInputSize = modelInputSize;

        foreach (var elem in elementMapping.Elements)
        {
            _classIdToElementType[elem.ClassId] = Enum.Parse<ElementType>(elem.Name);
        }
    }

    public bool LoadModel(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            Log.Error("YOLO model not found: {Path}", modelPath);
            return false;
        }

        try
        {
            var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            options.AppendExecutionProvider_CUDA(0);
            _session = new InferenceSession(modelPath, options);
            Log.Information("YOLO ONNX model loaded: {Path}", modelPath);
            return true;
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("cuda"))
        {
            Log.Warning("CUDA not available, falling back to CPU: {Msg}", ex.Message);
            try
            {
                var options = new SessionOptions();
                options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                _session = new InferenceSession(modelPath, options);
                Log.Information("YOLO ONNX model loaded (CPU): {Path}", modelPath);
                return true;
            }
            catch (Exception cpuEx)
            {
                Log.Error(cpuEx, "Failed to load YOLO model on CPU: {Path}", modelPath);
                return false;
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to load YOLO model: {Path}", modelPath);
            return false;
        }
    }

    public Dictionary<ElementType, DetectedElement> Detect(Bitmap bitmap, float confidenceThreshold)
    {
        var result = new Dictionary<ElementType, DetectedElement>();

        if (_session == null || bitmap == null)
            return result;

        try
        {
            var inputName = _session.InputNames[0];
            var inputTensor = PreprocessImage(bitmap);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var outputs = _session.Run(inputs);
            var output = outputs.First().AsTensor<float>();
            var detections = ParseDetections(output, confidenceThreshold, bitmap.Width, bitmap.Height);

            var nmsResults = NonMaxSuppression(detections, 0.45f);

            foreach (var det in nmsResults)
            {
                if (_classIdToElementType.TryGetValue(det.ClassId, out var elemType))
                {
                    if (!result.ContainsKey(elemType))
                    {
                        result[elemType] = new DetectedElement(
                            elemType,
                            det.ClassId,
                            (int)det.X1,
                            (int)det.Y1,
                            (int)det.X2,
                            (int)det.Y2);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "YOLO detection error");
        }

        return result;
    }

    private DenseTensor<float> PreprocessImage(Bitmap bitmap)
    {
        using var resized = new Bitmap(_modelInputSize, _modelInputSize);
        using var g = Graphics.FromImage(resized);
        g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
        g.DrawImage(bitmap, 0, 0, _modelInputSize, _modelInputSize);

        var bmpData = resized.LockBits(
            new Rectangle(0, 0, _modelInputSize, _modelInputSize),
            ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

        var pixelCount = _modelInputSize * _modelInputSize * 3;
        var rgbPixels = new byte[pixelCount];
        Marshal.Copy(bmpData.Scan0, rgbPixels, 0, pixelCount);
        resized.UnlockBits(bmpData);

        var tensor = new DenseTensor<float>(new[] { 1, 3, _modelInputSize, _modelInputSize });
        for (int y = 0; y < _modelInputSize; y++)
        {
            for (int x = 0; x < _modelInputSize; x++)
            {
                var srcIdx = (y * _modelInputSize + x) * 3;
                tensor[0, 0, y, x] = rgbPixels[srcIdx + 2] / 255.0f;
                tensor[0, 1, y, x] = rgbPixels[srcIdx + 1] / 255.0f;
                tensor[0, 2, y, x] = rgbPixels[srcIdx] / 255.0f;
            }
        }

        return tensor;
    }

    private List<Detection> ParseDetections(Tensor<float> output, float confThreshold, int imageWidth, int imageHeight)
    {
        var detections = new List<Detection>();
        var dimensions = output.Dimensions;

        int numDetections;
        int numClasses;
        bool transposed;

        if (dimensions.Length == 3)
        {
            if (dimensions[1] > dimensions[2])
            {
                numDetections = dimensions[2];
                numClasses = dimensions[1] - 4;
                transposed = true;
            }
            else
            {
                numDetections = dimensions[1];
                numClasses = dimensions[2] - 4;
                transposed = false;
            }
        }
        else if (dimensions.Length == 2)
        {
            numDetections = dimensions[0];
            numClasses = dimensions[1] - 4;
            transposed = false;
        }
        else
        {
            return detections;
        }

        float scaleX = (float)imageWidth / _modelInputSize;
        float scaleY = (float)imageHeight / _modelInputSize;

        for (int i = 0; i < numDetections; i++)
        {
            float cx, cy, w, h;
            int classId = -1;
            float classConf = 0f;

            if (transposed)
            {
                cx = output[0, i, 0];
                cy = output[0, i, 1];
                w = output[0, i, 2];
                h = output[0, i, 3];

                for (int c = 0; c < numClasses; c++)
                {
                    var score = output[0, i, 4 + c];
                    if (score > classConf)
                    {
                        classConf = score;
                        classId = c;
                    }
                }
            }
            else
            {
                cx = output[0, i, 0];
                cy = output[0, i, 1];
                w = output[0, i, 2];
                h = output[0, i, 3];

                for (int c = 0; c < numClasses; c++)
                {
                    var score = output[0, i, 4 + c];
                    if (score > classConf)
                    {
                        classConf = score;
                        classId = c;
                    }
                }
            }

            if (classConf < confThreshold || classId < 0)
                continue;

            var x1 = (cx - w / 2) * scaleX;
            var y1 = (cy - h / 2) * scaleY;
            var x2 = (cx + w / 2) * scaleX;
            var y2 = (cy + h / 2) * scaleY;

            detections.Add(new Detection
            {
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                Confidence = classConf,
                ClassId = classId
            });
        }

        return detections;
    }

    private static List<Detection> NonMaxSuppression(List<Detection> detections, float iouThreshold)
    {
        var result = new List<Detection>();
        if (detections.Count == 0) return result;

        var sorted = detections.OrderByDescending(d => d.Confidence).ToList();
        var suppressed = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (suppressed[i]) continue;

            result.Add(sorted[i]);

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;
                if (sorted[i].ClassId != sorted[j].ClassId) continue;

                if (ComputeIoU(sorted[i], sorted[j]) > iouThreshold)
                    suppressed[j] = true;
            }
        }

        return result;
    }

    private static float ComputeIoU(Detection a, Detection b)
    {
        var x1 = Math.Max(a.X1, b.X1);
        var y1 = Math.Max(a.Y1, b.Y1);
        var x2 = Math.Min(a.X2, b.X2);
        var y2 = Math.Min(a.Y2, b.Y2);

        var intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        if (intersection <= 0) return 0;

        var areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        var areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);
        var union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    private struct Detection
    {
        public float X1, Y1, X2, Y2, Confidence;
        public int ClassId;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
