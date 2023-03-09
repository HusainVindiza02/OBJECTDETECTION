using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Configuration;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using System.IO;

namespace VINDIZA
{
    public partial class Form1 : Form
    {
        VideoCapture videoCapture = null;
        string[] CocoClasses;
        Net CaffeModel=null;
        public Form1()
        {
            InitializeComponent();
        }

        private void detectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if(videoCapture==null)
                {
                    throw new Exception("загрузите видео");
                }

                var modelpath = @"C:\maskrcnn\frozen_inference_graph.pb";
                var configpath = @"C:\maskrcnn\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
                var coconames = @"C:\maskrcnn\coco-labels-paper.names";

                CaffeModel = DnnInvoke.ReadNetFromTensorflow(modelpath, configpath);
                CaffeModel.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                CaffeModel.SetPreferableTarget(Target.Cpu);

                CocoClasses = File.ReadAllLines(coconames);

                videoCapture.ImageGrabbed += processObjectDetection;
                videoCapture.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void processObjectDetection(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat();
                videoCapture.Read(frame);
                if (frame == null)
                {
                    return;
                }

                var img = frame.ToImage<Bgr, byte>();
                var blob = DnnInvoke.BlobFromImage(img, 1.0, frame.Size, swapRB: true);
                CaffeModel.SetInput(blob);

                var output = new VectorOfMat();
                string[] outnames = new string[] { "detection_out_final" };
                CaffeModel.Forward(output, outnames);

                var threshold = 0.6;

                int numDetections = output[0].SizeOfDimension[2];
                int numClasses = 90;

                var bboxes = output[0].GetData();

                for(int i=0; i < numDetections; i++)
                {
                    float score = (float) bboxes.GetValue(0,0,i,2);
                    if (score<threshold)
                    {
                        int classId = Convert.ToInt32(bboxes.GetValue(0,0,i,1));
                        int left = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 3)*img.Cols);
                        int top = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 4) * img.Rows);
                        int right = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 5) * img.Cols);
                        int bottom = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 6) * img.Rows);

                        Rectangle rectangle = new Rectangle(left, top, right-left+1, bottom-top+1);
                        img.Draw(rectangle, new Bgr(0, 0, 255), 2);
                        var labels = CocoClasses[classId];
                        CvInvoke.PutText(img,labels,new Point(left,top-10), FontFace.HersheySimplex,1.0, new MCvScalar(0,255,0),2);
                    }
                }
                pictureBox2.Invoke((MethodInvoker)delegate { pictureBox2.Image = img.ToBitmap(); });
                
            }
            catch(Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }
        private void openVideoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog= new OpenFileDialog();
                dialog.Filter = "Video Files(*.avi,*.mp4;)|*.avi;*.mp4;|All Files(*.*;)|*.*;";

                if(dialog.ShowDialog()== DialogResult.OK)
                {
                    if (videoCapture!=null && videoCapture.IsOpened)
                    {
                        videoCapture.Dispose();
                        videoCapture = null;
                    }
                    videoCapture= new VideoCapture(dialog.FileName);
                    Mat frame = new Mat();
                    videoCapture.Read(frame);

                    pictureBox2.Image = frame.Bitmap;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
    }
}
