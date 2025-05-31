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
import org.opencv.core.Rect;
import org.opencv.core.Point;
import java.util.ArrayList;
import org.openftc.easyopencv.OpenCvPipeline;
public class OrangeTracker extends OpenCvPipeline {

	// Color and line thickness for drawing lines on contours
	public Scalar lineColor = new Scalar(0.0, 255.0, 0.0, 0.0);
	public int lineThickness = 3;

	//lower and upper values of colors to search for
	public Scalar lowerHSV = new Scalar(6.0, 98.0, 98.0, 0.0);
	public Scalar upperHSV = new Scalar(28.0, 255.0, 255.0, 0.0);

	//min_area for a contour to count as an object
	public double MIN_AREA = 900;


	private Mat hsvMat = new Mat(); // Matrix of hsv image
	private Mat hsvBinaryMat = new Mat(); // Matrix of binary image (0-1 black and white)

	private ArrayList<MatOfPoint> contours = new ArrayList<>(); // List of detected contours
	private Mat hierarchy = new Mat(); // idk 

	private MatOfPoint biggestContour = null; // Reference to biggest contour

	private Mat inputContours = new Mat(); // original image

	double objWidth = 7.62; // width of gameObject in cm
	double focalLength = 1015.5; // focal length of mac camera in pixels

	private Telemetry telemetry;


	public OrangeTracker(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

	@Override
	public Mat processFrame(Mat input) {
		Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_RGB2HSV); // convert RGB img to HSV
		Core.inRange(hsvMat, lowerHSV, upperHSV, hsvBinaryMat); // create binary mask of detected color

		contours.clear();
		hierarchy.release();

		// find contours
		Imgproc.findContours(hsvBinaryMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


		this.biggestContour = null;
		double bestAspectRatio = 99999;
		
		// loop through all found contours
		for(MatOfPoint contour : contours) {
			// Get bounding rectangle of each contour
			Rect boundingRect = Imgproc.boundingRect(contour); 
			double aspectRatio = (double) boundingRect.width / (double) boundingRect.height;
			double aspectRatioDiff = Math.abs(aspectRatio - 1.0);
			double area = (double) boundingRect.width * (double) boundingRect.height;

			//check if contour has best aspect ratio and good area
			if (aspectRatioDiff < bestAspectRatio && area > MIN_AREA) {
                bestAspectRatio = aspectRatioDiff;
                this.biggestContour = contour;
            }
		}

		//copy the input image to draw contours on
		input.copyTo(inputContours);

		ArrayList<MatOfPoint> contoursList = new ArrayList<>();
		if(biggestContour != null) {
			contoursList.add(biggestContour);
			Rect boundingRect = Imgproc.boundingRect(biggestContour);
			double objectWidthInPixels = boundingRect.width;

			// Calculate distance
			double distance = (objWidth * focalLength) / objectWidthInPixels;

			telemetry.addData("Rough Distance (cm)", distance);
			telemetry.update();
			// Draw text on the image
			String distanceText = String.format("Distance: %.2f cm", distance);
			// Position the text near the detected object
			Point textPosition = new Point(boundingRect.x, boundingRect.y - 10); // Adjust position as needed
			Imgproc.putText(inputContours, distanceText, textPosition, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
		}

		Imgproc.drawContours(inputContours, contoursList, -1, lineColor, lineThickness);

		return inputContours;
	}
}
