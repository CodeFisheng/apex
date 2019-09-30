import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class DecompConv2d(Function):
    def __init__(self):
        super(DecompConv2d, self).__init__()

    @staticmethod
    def forward(cxt, input, weight, avg_buffer, stride=1, padding=0, group=1):
        # print('DecompConv2d Forward. ')
        # print(input.shape)
        new_input = torch.mean(input.view(input.shape[0]//group, group, input.shape[1], input.shape[2], input.shape[3]), dim=0)#.contiguous().view(group, input.shape[1], input.shape[2], input.shape[3])
        # print(new_input.shape)
        # assert(0==1)
        eta = 0.997
        if torch.eq(avg_buffer, 0).all() == True:
            avg_buffer.data = new_input
        else:
            delta = (1-eta) * (new_input-avg_buffer)
            avg_buffer.data += delta
        cxt.save_for_backward(new_input, weight)
        # cxt.save_for_backward(input, weight)
        cxt.stride = stride
        cxt.padding = padding
        cxt.batchsize = input.shape[0]
        cxt.group = group
        # print('out shape: ', out.shape)
        return F.conv2d(input, weight, stride=stride, padding=padding)

    @staticmethod
    def backward(cxt, grad_outputs):
        # print('DecompConv2d Backward. ')
        input, weight = cxt.saved_tensors
        stride = cxt.stride
        padding = cxt.padding
        batchsize = cxt.batchsize
        group = cxt.group
        # print(input.shape)
        # new_input = input.contiguous().repeat(batchsize, 1, 1, 1)
        # print(new_input.shape)
        # assert(0==1)
        new_grad_outputs = torch.sum(grad_outputs.view(grad_outputs.shape[0]//group, group, grad_outputs.shape[1], grad_outputs.shape[2], grad_outputs.shape[3]), dim=0)#.contiguous().view(group, grad_outputs.shape[1], grad_outputs.shape[2], grad_outputs.shape[3])
        grad_input = F.grad.conv2d_input([batchsize, input.shape[1], input.shape[2], input.shape[3]], 
                                         weight, grad_outputs, stride=stride, padding=padding)
        # new_input = nn.AvgPool2d((2, 2), stride=(2, 2))(input)
        # new_grad_outputs = 4.2 * nn.AvgPool2d((2, 2), stride=(2, 2))(grad_outputs)
        # print('hehe')
        # print(input.shape)
        # print(new_grad_outputs.shape)
        # assert(0==1)
        grad_weight = F.grad.conv2d_weight(input, weight.shape, new_grad_outputs, stride=stride, padding=padding)
        # del new_input
        return grad_input, grad_weight, None, None, None, None

def main():
    print('-------- demo --------')
    weight = torch.Tensor(np.random.randn(1, 1, 2, 2))

    x1 = torch.Tensor(np.random.randn(1, 1, 2, 2))
    x1.requires_grad = True
    weight.requires_grad = True
    conv1 = DecompConv2d.apply
    y1 = conv1(x1, weight)
    loss = torch.mean(y1)
    loss.backward()
    print(x1.size(), x1)
    print('y1: ', y1)
    conv = nn.Conv2d(1, 1, 2)
    y1 = conv(x1)
    print('w: ', weight)
    print(y1)


if __name__ == '__main__':
    main()
