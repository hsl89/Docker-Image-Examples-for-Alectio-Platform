from cocodata import trainset, testset
from utils.utils import *
from torch.utils.data import SubsetRandomSampler, DataLoader
from hyperparameters import hyp
import torch.optim as optim
import torch
from samplers import SubsetSampler

from model import Darknet

DEVICE = os.getenv('CUDA_DEVICE')
TASK = os.getenv('TASK')
CKPT_FILE = os.getenv('CKPT_FILE')
RESUME_FROM = os.getenv('RESUME_FROM')
EXPT_DIR = os.getenv('EXPT_DIR')
CUDA_DEVICE = os.getenv('CUDA_DEVICE')


class Flow:
    def __init__(self, model, dataset, testset, hyp, resume_from=None):
        self.model = model
        self.dataset = dataset
        self.testset = testset
        self.device = CUDA_DEVICE

        
        pg0, pg1, pg2 = [], [], []
        for k,v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2+=[v]
            elif 'Conv2d.weight' in k:
                pg1+=[v]
            else:
                pg0+=[v]
        self.optimizer = optim.Adam(pg0,lr=hyp['lr0'])
        self.optimizer.add_param_group({'params': pg1, 
            'weight_decay': hyp['weight_decay']})
        self.optimizer.add_param_group({'params': pg2})
        del pg0, pg1, pg2

        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        self.model.nc = 80 
        self.model.arc = 'default' # default yolo arch
        self.model.hyp = hyp
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, 1).to(
                self.device)

        self.model.to(self.device)

    def train(self, epoch, samples=None, batch_size=16, prebias=True):
        if prebias:
            if epoch < 3:
                ps = 0.1, 0.9
            else:
                ps = hyp['lr0'], hyp['momentum']
                prebias=False
            self.optimizer.param_groups[2]['lr'] = ps[0]


        if samples:
            sampler = SubsetRandomSampler(samples)
        else:
            sampler = SubsetRandomSampler(range(len(self.dataset)))

        self.dataset.augment=True # augment while training
        dataloader = DataLoader(self.dataset, batch_size=batch_size,
                sampler=sampler, collate_fn=self.dataset.collate_fn)
        self.model.train()
        mloss = torch.zeros(4).to(self.device)
        loss_titles = ['giou', 'obj', 'cls', 'total']

        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.float() / 255.0
            imgs, targets = imgs.to(self.device), targets.to(self.device)

            pred = self.model(imgs)
            loss, loss_items = compute_loss(pred, targets, self.model, 
                    not prebias)
            
            if not torch.isfinite(loss):
                print('Warning: non-finite loss, skipping this batch')
                continue

            loss *= batch_size / 64
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            if i % 10 == 0:
                print(i, *zip(loss_titles, mloss.cpu().numpy().tolist()))

        # save ckpt
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        torch.save(ckpt, os.path.join(EXPT_DIR, CKPT_FILE))

        return tuple(mloss.cpu().numpy().tolist())

    def validate(self, samples=None, batch_size=32, 
            conf_thres=0.1, iou_thres=0.6):
        # num of class
        nc = 80
        seen = 0
        verbose=True
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
        iouv = iouv[0].view(1)
        niou = iouv.numel()
        jdict, stats, ap, ap_class = [], [], [], []
        self.model.eval()
        if not samples:
            samples = range(len(self.testset))
        
        sampler = SubsetSampler(samples)

        dataloader = DataLoader(
            self.testset, batch_size=batch_size, 
            sampler=sampler,
            collate_fn=self.testset.collate_fn)
        
        prediction = []
        for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            with torch.no_grad():
                imgs = imgs.float() / 255.0
                _, _, height, width = imgs.shape

                imgs, targets = imgs.to(self.device), targets.to(self.device)
                inf_out, _ = self.model(imgs)
                
                inf_out = inf_out.cpu()
                
                # run nms
                print('running nms')
                # list of bbox on each image
                bboxs = non_max_suppression(inf_out, conf_thres=conf_thres,
                        iou_thres=iou_thres)
                prediction.extend(bboxs)


        for i, bbox in enumerate(prediction):
            if bbox is not None:
                prediction[i] = bbox.numpy().tolist()
            else:
                prediction[i] = []

        prediction = {samples[i]: prediction[i] for i in range(len(samples))}

        # save prediction to EXPT_DIR
        with open(os.path.join(EXPT_DIR, 'prediction.pkl'), 'wb') as f:
            pickle.dump(prediction, f)
        
        return 

    def infer(self, samples, batch_size=16):
        '''infer on unlabeled samples'''
        self.model.eval()
        self.dataset.augment=False # no augment while infering
        sampler = SubsetSampler(samples)
        dataloader = DataLoader(self.dataset, batch_size=batch_size,
                sampler=sampler, collate_fn=self.dataset.collate_fn)

        output = []
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.float() / 255.0
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                inf_out, _ = self.model(imgs)
                inf_out = inf_out.cpu()
                for t in inf_out:
                    output.append(t.numpy().tolist())
        output = {samples[i]: output[i] for i in range(len(samples))} 
        with open(os.path.join(EXPT_DIR, 'output.pkl'), 'wb') as f:
            pickle.dump(output, f)


if __name__=='__main__':
    import os
    model = Darknet() 
    fw = Flow(model, trainset, testset, hyp, 
            resume_from=os.path.join(
                os.getenv('HOME'), 'Common/COCOyolov3/B/ckpt_49.pt'))


    x = fw.infer(range(10))
