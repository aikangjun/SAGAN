import tensorflow as tf
from network.Generator import Generator
import configure.config as cfg
import tensorflow.keras as keras

layers = keras.layers
models = keras.models
if __name__ == '__main__':
    # 以下是将训练好的模型转为SavedModel格式模型(也叫tf模型)
    model_g = Generator()
    ckpt = tf.train.Checkpoint(g=model_g)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=cfg.ckpt_path,
                                              max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('最新检测点已加载')
    input = layers.Input(shape=(100,))
    output = model_g(input)
    m = models.Model(input, output)
    # m.save(filepath='.\\tf_model_',save_format='tf')
    tf.saved_model.save(m, '.\\tf_model')

    # 将saved_model转为onnx需要在terminal的cmd运行下面指令
    # python -m tf2onnx.convert --saved-model ./tf_model/ --output ./models/model.onnx
