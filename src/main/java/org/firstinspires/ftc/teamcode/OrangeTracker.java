package org.firstinspires.ftc.teamcode;

import android.graphics.Canvas;
import com.qualcomm.robotcore.eventloop.opmode.Disabled;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint;
import java.util.ArrayList;
import org.openftc.easyopencv.OpenCvPipeline;

public class OrangeTracker extends OpenCvPipeline {

	public Scalar lineColor = new Scalar(0.0, 255.0, 0.0, 0.0);
	public int lineThickness = 3;

	public Scalar lowerHSV = new Scalar(6.0, 98.0, 98.0, 0.0);
	public Scalar upperHSV = new Scalar(28.0, 255.0, 255.0, 0.0);
	private Mat hsvMat = new Mat();
	private Mat hsvBinaryMat = new Mat();

	private ArrayList<MatOfPoint> contours = new ArrayList<>();
	private Mat hierarchy = new Mat();

	private MatOfPoint biggestContour = null;

	private Mat inputContours = new Mat();

	@Override
	public Mat processFrame(Mat input) {
		Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_RGB2HSV);
		Core.inRange(hsvMat, lowerHSV, upperHSV, hsvBinaryMat);

		contours.clear();
		hierarchy.release();
		Imgproc.findContours(hsvBinaryMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

		this.biggestContour = null;
		for(MatOfPoint contour : contours) {
			if((biggestContour == null) || (Imgproc.contourArea(contour) > Imgproc.contourArea(biggestContour))) {
				this.biggestContour = contour;
			}
		}

		input.copyTo(inputContours);

		ArrayList<MatOfPoint> contoursList = new ArrayList<>();
		if(biggestContour != null) {
			contoursList.add(biggestContour);
		}

		Imgproc.drawContours(inputContours, contoursList, -1, lineColor, lineThickness);

		return inputContours;
	}
}
