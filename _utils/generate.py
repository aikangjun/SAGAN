import cv2
import numpy as np
import glob


class DataGenerator():
    '''
    使用二次元头像作为数据集，编写generator
    '''

    def __init__(self,
                 img_path: str,
                 batch_size: int,
                 image_size: tuple):

        self.img_path = img_path
        self.batch_size = batch_size
        self.image_size = image_size

        self.files = glob.glob(self.img_path + '\\*')

    def get_train_step(self):
        if len(self.files) % self.batch_size == 0:
            return len(self.files) // self.batch_size
        else:
            return len(self.files) // self.batch_size + 1

    def generate(self):
        files = self.files
        while True:
            real_images = []
            for i, file in enumerate(files):
                img = cv2.imread(file)
                assert img.shape[-1] == 3
                img = cv2.resize(img, dsize=self.image_size, interpolation=cv2.INTER_CUBIC)
                img = img / 127.5 - 1
                real_images.append(img)
                if len(real_images) % self.batch_size == 0 or i - 1 == len(self.files):
                    anno_real_images = np.array(real_images, dtype='float32')
                    real_images.clear()
                    yield anno_real_images


if __name__ == '__main__':
    import configure.config as cfg

    gen = DataGenerator(img_path=cfg.file_path, batch_size=6, image_size=cfg.image_size)
    train_gen = gen.generate()
    real_images = next(train_gen)
    cv2.imshow('1', real_images[0])
    cv2.waitKey(0)
