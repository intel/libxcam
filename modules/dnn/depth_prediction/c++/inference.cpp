#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (argc != 3) {
            cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image>" << endl;
            return EXIT_FAILURE;
        }

        const string input_model{ argv[1] };
        const string input_image_path{ argv[2] };
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        ov::Core ie;
        vector<string> availableDevices = ie.get_available_devices();
        for (int i = 0; i < availableDevices.size(); i++)
            printf("supported device name:%s\n", availableDevices[i].c_str());
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 2. Read a model in OpenVINO Intermediate Representation (.xml files)
        // -------------------------------------
        ov::CompiledModel compiled_model = ie.compile_model(input_model, "AUTO");
        printf("Successfully loaded the model\n");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Create an infer request
        // -------------------------------------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Get input info
        // -------------------------------------
        ov::Tensor input_tensor = infer_request.get_input_tensor();
        ov::Shape tensor_shape = input_tensor.get_shape();
        size_t feed_channels = tensor_shape[1];
        size_t feed_height = tensor_shape[2];
        size_t feed_width = tensor_shape[3];
        // --------------------------- Step 5. Prepare input
        // -------------------------------------
        Mat src = imread(input_image_path, COLOR_BGR2RGB);
        int original_width = src.cols;
        int original_height = src.rows;
        Mat blob_image;
        resize(src, blob_image, Size(feed_width, feed_height), INTER_LANCZOS4);
        blob_image = blob_image / 255.0;
        float* image_data = input_tensor.data<float>();
        for (size_t channels = 0; channels < feed_channels; channels++)
            for (size_t row = 0; row < feed_height; row++)
                for (size_t col = 0; col < feed_width; col++) {
                    image_data[channels * feed_height * feed_width + row * feed_width + col] =
                        (float)blob_image.at<Vec3b>(row, col)[channels];
                }
        // --------------------------- Step 6. Do inference
        // --------------------------------------
        infer_request.infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Process output
        // --------------------------------------
        auto output_tensor = infer_request.get_output_tensor();
        const float* detection = (float*)output_tensor.data();
        ov::Shape out_shape = output_tensor.get_shape();
        size_t out_c = out_shape[1];
        size_t out_h = out_shape[2];
        size_t out_w = out_shape[3];
        Mat result = Mat::zeros(Size(out_w, out_h), CV_32FC1);
        for (int row = 0; row < out_h; row++)
            for (int col = 0; col < out_w; col++) {
                result.at<float>(row, col) =
                    float(detection[row * out_w + col]);
            }
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(result, minp, maxp);
        Mat out_img;
        resize(result, out_img, Size(original_width, original_height), 0, 0, INTER_LINEAR);
        out_img.convertTo(out_img, CV_8U, 255.0 / (maxv - minv));
        Mat im_color;
        applyColorMap(out_img, im_color, COLORMAP_MAGMA);
        imwrite("pred_disp.jpg", im_color);
        printf("Successfully saved the image\n");
    }
    catch (const exception& ex) {
        cerr << ex.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

