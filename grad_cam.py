import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import nibabel as nib
from mean_pertrub import MeanPertrub

class GuidedBackprop():
    def __init__(self, model):
        self.model = model

    def guided_backprop(self, input, label):

        def hookfunc(module, gradInput, gradOutput):
            return tuple([(None if g is None else g.clamp(min=0)) for g in gradInput])

        input.requires_grad = True
        h = [0] * len(list(self.model.features) + list(self.model.classifier))
        for i, module in enumerate(list(self.model.features) + list(self.model.classifier)):
            if type(module) == nn.ReLU:
                h[i] = module.register_backward_hook(hookfunc)

        self.model.eval()
        output = self.model(input)
        self.model.zero_grad()
        output[0][label].backward()
        grad = input.grad.data
        grad /= grad.max()
        return np.clip(grad.cpu().numpy(), 0, 1)


class AttentionMap():
    def __init__(self, model):
        self.model = model

    def guided_backprop(self, input):
        input.requires_grad = True
        self.model.eval()
        act = self.model.get_activations(input)
        attention = []
        for c in range(act.shape[1]):
            activation = self.model.get_activations(input)
            activation[0, c].backward(torch.ones_like(activation[0, c]))
            attention.append(input.grad.data.cpu().numpy().squeeze(0))
            self.model.zero_grad()
        return np.concatenate(attention, axis=0)


def get_masks(model, loader, fold, output_dir, mean_mask = True, mask_type='grad_cam', size=(180, 180, 180), task = 'test', save = None, save_binary=None):
    masks = []
    labels = []
    mask_dir = os.path.join(output_dir, 'fold-%i' % fold, 'img_mask_{}'.format(task))
    os.makedirs(mask_dir, exist_ok=True)
    for i, data in tqdm(enumerate(loader, 0)):
        image = data['image'].cuda()
        labels.append(data['label'].numpy().item())
        logit = model(image)
        if mask_type == 'grad_cam':
            logit[:, logit.data.max(1)[1]].backward()
            #             logit[:,0].backward()
            activation = model.get_activations(image).detach()
            act_grad = model.get_activations_gradient()
            pool_act_grad = torch.mean(act_grad, dim=[2, 3, 4], keepdim=True)
            activation = activation * pool_act_grad
            heatmap = torch.sum(activation, dim=1)
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)
            heatmap = F.interpolate(heatmap.unsqueeze(0), size[1:], mode='trilinear', align_corners=False)  # 58 70 58
            masks.append(heatmap.cpu().numpy())
            name = data['image_path'][0][-80:-53]
            if save:
                nib.save(nib.Nifti1Image(heatmap.cpu().numpy(), affine=np.eye(4)),
                     os.path.join(mask_dir, '{}_gradcam_mask.nii.gz'.format(name)))
            if save_binary:
                mask_binary_dir = os.path.join(output_dir, 'fold-%i' % fold, 'img_mask_binary')
                os.makedirs(mask_dir, exist_ok=True)
                binary = heatmap.cpu().numpy()[heatmap.cpu().numpy() <= 0.35] = 0
                nib.save(nib.Nifti1Image(binary, affine=np.eye(4)),
                         os.path.join(mask_binary_dir, '{}_gradcam_mask_label{}.nii.gz'.format(name, data['label'].numpy().item())))
            del image, heatmap, activation, act_grad, pool_act_grad
        elif mask_type == 'guided_backprop':
            gp = GuidedBackprop(model)
            pred = logit.data.max(1)[1].item()
            img_grad = gp.guided_backprop(image, pred)
            masks.append(img_grad)
            del image, img_grad, pred
        elif mask_type == 'mean_pertrub':
            mp = MeanPertrub(rep=9)
            pred = logit.data.max(1)[1].item()
            for param in model.parameters():
                param.requires_grad = False
            masks_pertrub = mp.get_masks(image, pred, model)
            masks.append(masks_pertrub)
            del image, masks_pertrub, pred
        else:
            raise NotImplementedType('define mask_type')
    if mean_mask:
            name = data['image_path'][0][-80:-53]
            concat = np.concatenate(masks, axis=0).squeeze(axis=1)
            labels_cn = np.array(labels) == 0
            labels_ad = np.array(labels) == 1
            mean_0 = concat[labels_cn].mean(axis=0)
            mean_1 = concat[labels_ad].mean(axis=0)
            m_dir = os.path.join(output_dir, 'fold-%i' % fold)
            nib.save(nib.Nifti1Image(mean_0, affine=np.eye(4)),
                     os.path.join(m_dir, '{}_{}_mean_0_{}.nii.gz'.format(name, mask_type, task)))
            nib.save(nib.Nifti1Image(mean_1, affine=np.eye(4)),
                     os.path.join(m_dir, '{}_{}_mean_1_{}.nii.gz'.format(name, mask_type, task)))
