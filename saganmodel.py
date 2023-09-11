import math
import numpy as np
from network.Generator import Generator
from network.Discrminator import Discriminator
import tensorflow as tf
import tensorflow.keras as keras
import cv2
from PIL import Image
import os
import configure.config as cfg


class SAGANModlel():
    def __init__(self,
                 lr: float,
                 **kwargs):
        super(SAGANModlel, self).__init__(**kwargs)

        self.g = Generator()
        self.d = Discriminator()

        # 对判别器和生成器使用不同的学习速度。使用较低的学习率更新生成器，判别器使用较高的学习率进行更新。
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
        self.optimizer_d = keras.optimizers.Adam(learning_rate=lr * 3, beta_1=0.0)
        self.optimizer_g = keras.optimizers.Adam(learning_rate=lr, beta_1=0.0)
        self.train_loss_d = keras.metrics.Mean()
        self.train_loss_g = keras.metrics.Mean()

    @tf.function
    def train_step(self, random_noise, real_images):
        # 在同一个tape中同步更新discriminator和generator，计算速度会变快
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            fake_images = self.g(random_noise)

            real_score = self.d(real_images)
            fake_score = self.d(fake_images)

            real_loss = self.loss_fn(tf.ones_like(real_score), real_score)
            fake_loss = self.loss_fn(tf.zeros_like(fake_score), fake_score)
            d_loss = tf.concat([real_loss, fake_loss], axis=-1)
            g_loss = self.loss_fn(tf.ones_like(fake_score), fake_score)
        gradients_d = tape_d.gradient(d_loss, self.d.trainable_variables)
        gradients_g = tape_g.gradient(g_loss, self.g.trainable_variables)
        self.optimizer_d.apply_gradients(zip(gradients_d, self.d.trainable_variables))
        self.optimizer_g.apply_gradients(zip(gradients_g, self.g.trainable_variables))
        self.train_loss_d(d_loss)
        self.train_loss_g(g_loss)

    @staticmethod
    def merge(fake_images, gap=4):
        fake_images = (fake_images + 1) * 127.5
        fake_images = np.array(fake_images, dtype=np.uint8)
        fake_images = np.clip(fake_images, 0, 255)
        fake_images = [Image.fromarray(img) for img in fake_images]
        w, h = fake_images[0].size
        newimg_len = int(math.sqrt(len(fake_images)))
        newimg = Image.new(fake_images[0].mode,
                           size=((w + gap) * newimg_len - gap, (h + gap) * newimg_len - gap),
                           color=(255, 255, 255))
        i = 0
        for row in range(newimg_len):
            for col in range(newimg_len):
                newimg.paste(fake_images[i], box=((w + gap) * row, (h + gap) * col))
                i = i + 1
        return np.array(newimg)

    def fake_image_save(self, path, num_imgs, epoch=0):
        assert num_imgs > -1 and (num_imgs ** 0.5 % 1 == 0), 'num_imgs必须是被开方数为整数'
        random_noise = tf.random.normal(shape=(num_imgs, 100))
        fake_images = self.g(random_noise)
        newimg = self.merge(fake_images)
        if not os.path.exists(cfg.result_path):
            os.makedirs(cfg.result_path)
        cv2.imwrite(path + f'\\epoch{epoch + 1}.jpg', newimg)


if __name__ == '__main__':
    g = Generator()
    random_noise = tf.random.normal(shape=(25, 100))
    fake_images = g(random_noise)
    model = SAGANModlel(lr=1e-3)
    model.fake_image_save(path=cfg.result_path, num_imgs=25)
