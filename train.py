import tensorflow as tf
from _utils.generate import DataGenerator
from saganmodel import SAGANModlel
import configure.config as cfg
import os

if __name__ == '__main__':
    gen = DataGenerator(img_path=cfg.file_path,
                        batch_size=cfg.batch_size,
                        image_size=cfg.image_size)
    model = SAGANModlel(lr=cfg.lr)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(g=model.g,
                               d=model.d,
                               optimizer_d=model.optimizer_d,
                               optimizer_g=model.optimizer_g)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=cfg.ckpt_path,
                                              max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('最新检测点已加载')

    train_gen = gen.generate()
    '''
    判别器的损失=0说明模型训练失败。
    如果生成器的损失稳步下降，说明判别器没有起作用。
    需要训练200次左右能够生成较清晰图像
    '''
    for epoch in range(cfg.epochs):
        for i in range(gen.get_train_step()):
            real_images = next(train_gen)
            random_noise = tf.random.normal(shape=(cfg.batch_size, 100))
            model.train_step(random_noise, real_images)
            if (i + 1) % 100 == 0:
                print(f'Batch {i + 1}:\t'
                      f'train_loss_d: {model.train_loss_d.result()}\t'
                      f'train_loss_g: {model.train_loss_g.result()}\n')

        print(f'Epoch {epoch + 1}:\t'
              f'train_loss_d: {model.train_loss_d.result()}\t'
              f'train_loss_g: {model.train_loss_g.result()}\n')
        model.fake_image_save(path=cfg.result_path, num_imgs=36, epoch=epoch)

        save_path = ckpt_manager.save()
        model.train_loss_d.reset_state()
        model.train_loss_g.reset_state()
