import onnxruntime as rt


class BasePredictor(object):
    def get_onnx_session(self, model_dir, use_gpu=False):
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        return rt.InferenceSession(model_dir, providers=providers)

    def get_output_name(self, session):
        return [node.name for node in session.get_outputs()]

    def get_input_name(self, session):
        return [node.name for node in session.get_inputs()]

    def get_input_feed(self, input_name, np_image):
        input_feed = {}
        for name in input_name:
            input_feed[name] = np_image
        return input_feed
