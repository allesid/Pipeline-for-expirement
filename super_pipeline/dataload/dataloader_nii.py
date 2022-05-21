import torch
import os
from super_pipeline.main import lib


class dataloader3D():
    def __init__(self, data_path, trvaldir, batch_size):
        # test_dir = data_path + "test"
        # data_path = "/run/media/alex/PartOfDiskWithWin7/LEARN/Longevity/Liver/datasets/wiki.cancerimagingarchive.net/Crowds Cure Cancer: Data collected at the RSNA 2017 annual meeting/manifest-KTt2tScD7164745271364348431/TCGA-LIHC/1-test/"
        sample_targets = sorted(os.listdir(
            os.path.join(data_path, trvaldir)), reverse=True)
        samples = sorted(os.listdir(os.path.join(data_path, trvaldir, sample_targets[1])))
        targets = sorted(os.listdir(os.path.join(data_path, trvaldir, sample_targets[0])))
        self.samples_path = os.path.join(
            data_path, trvaldir, sample_targets[1])
        self.targets_path = os.path.join(
            data_path, trvaldir, sample_targets[0])
        self.st = list(zip(targets, samples))
        self.batch_size = batch_size
        # nxt = 0
        self.k = 0


    def triple(self, i, t, s, sz):
        target = torch.zeros([sz, t.shape[1], t.shape[2]])
        sample = torch.zeros([sz, 3, s.shape[1], s.shape[2]])
        for j in range(sz):
            target[j, :, :] = t[i+j, :, :]
            sample[j, :, :, :] = torch.broadcast_to(
                s[i+j, :, :], (3,)+tuple(s[i+j, :, :].shape))
        return target, sample


    def dataloader(self):
        for self.k in range(len(self.st)):
            if self.k == 0:
                # k = 0
                nxt = 0
            if nxt == 0:
                ts = self.st[self.k]
                t = lib.conv_1img_3D_totensor(
                    os.path.join(self.targets_path, ts[0]))
                # print(os.path.join(self.targets_path, ts[0]))
                t = t / 2
                s = lib.conv_1img_3D_totensor(
                    os.path.join(self.samples_path, ts[1]))
                # print(os.path.join(self.samples_path, ts[1]))
                s = lib.img_normalize(s)
                print("1 : ", t.shape)
                print("2 : ", s.shape)
                tail = t.shape[0] % self.batch_size
                nmax = t.shape[0]//self.batch_size
            for nxt in range(nmax):
                target, sample = self.triple(nxt, t, s, self.batch_size)
                yield sample, target
            else:
                if tail > 0:
                    target, sample = self.triple(nmax, t, s, tail)
                    # print("tail=", tail, target.shape, sample.shape)
                nxt = 0
                self.k += 1
                del t
                del s
                yield sample, target
