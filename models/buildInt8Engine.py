import os
import cv2
import glob
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit


onnx_file_path="/home/yolo_v5_tensorrt/models/yolov5x.onnx"

# mode="FP32"
# engine_file_path="/home/yolo_v5_tensorrt/models/yolov5x_fp32.trt"

mode="INT8"
engine_file_path="/home/yolo_v5_tensorrt/models/yolov5x_int8.trt"
calibration_table_path="/home/yolo_v5_tensorrt/models/yolov5x_int8.cache"

calib_images_dir="/home/val2017/"

batch_size=8
width=640
height=640
calib_count=625



def Preprocess(input_img, width, height):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height)).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

class CalibDataLoader:
    def __init__(self, batch_size, width, height, calib_count, calib_images_dir):
        self.index = 0
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.calib_count = calib_count
        self.image_list = glob.glob(os.path.join(calib_images_dir, "*.jpg"))
        assert (
                len(self.image_list) >= self.batch_size * self.calib_count
        ), "{} must contains more than {} images for calibration".format(
            calib_images_dir, self.batch_size * self.calib_count
        )
        self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                assert os.path.exists(image_path), "image {} not found!".format(image_path)
                image = cv2.imread(image_path)
                image = Preprocess(image, self.width, self.height)
                self.calibration_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        # Moving calibration data from the CPU to the GPU
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        # If the calibration file exists, read the calibration file directly
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # If calibration is performed, the calibration table is written to the file for next use.
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()

def build_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    assert os.path.exists(onnx_file_path), "The onnx file {} is not found".format(onnx_file_path)
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("Building an engine from file {}, this may take a while...".format(onnx_file_path))

    # build tensorrt engine
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * (1 << 30))
    if mode == "INT8":
        print("TensorRT Engine Quant Mode Set as INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        data_loader = CalibDataLoader(batch_size, width, height, calib_count, calib_images_dir)
        calibrator = Calibrator(data_loader, calibration_table_path)
        config.int8_calibrator = calibrator
    elif mode == "FP32":
        print("TensorRT Engine precision FP32")
    else:
        print("Only Quant Mode Set Error")

    engine = builder.build_engine(network, config)

    if engine is None:
        print("Failed to create the engine")
        return None
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

    return engine


if __name__ == "__main__":
    build_engine()