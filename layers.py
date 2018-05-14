"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        #############################################################
        # code here
        outputs = np.dot(inputs, self.weights)+self.bias
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        batch, out_features = in_grads.shape
        self.b_grad=np.sum(in_grads, axis=0)
        out_grads = np.zeros_like(inputs)
        for b in range(batch):
            self.w_grad += np.outer(inputs[b,:], in_grads[b,:])
            for i in range(out_features):
                out_grads[b,:] += in_grads[b,i]*self.weights[:,i]
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        # code here
        batch, in_channel, in_height, in_width = inputs.shape
        x_hat = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        x_hat[ :, :,self.pad:in_height+self.pad, self.pad:in_width+self.pad]=inputs
        out_height, out_width= [int((in_height+2*self.pad-self.kernel_h)/self.stride)+1, int((in_width+2*self.pad-self.kernel_w)/self.stride)+1]
        outputs = np.zeros((batch, self.out_channel, out_height, out_width))
        for b in range(batch):
            for f in range(self.out_channel):
                for (w,h) in [(w,h) for w in range(out_width) for h in range(out_height)]:
                    receiptive_field=x_hat[b,:,(h*self.stride):(h*self.stride+self.kernel_h),(w*self.stride):(w*self.stride+self.kernel_w)]
                    outputs[b,f,h,w] =np.sum(receiptive_field*self.weights[f,:,:,:])+self.bias[f]
        
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        batch, in_channel, in_height, in_width = inputs.shape
        batch, out_channel, out_height, out_width = in_grads.shape
        out_grads = np.zeros((batch, in_channel, in_height, in_width))
        self.b_grad=np.sum(in_grads, axis=(0,2,3))
        x_hat = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        x_hat[ :, :,self.pad:in_height+self.pad, self.pad:in_width+self.pad]=inputs
        for b in range(batch):
            for f in range(self.out_channel):
                for (w,h) in [(w,h) for w in range(out_width) for h in range(out_height)]:
                    receiptive_field=x_hat[b,:,(h*self.stride):(h*self.stride+self.kernel_h),(w*self.stride):(w*self.stride+self.kernel_w)]
                    self.w_grad[f,:,:,:]+=receiptive_field*in_grads[b,f,h,w]
                    out_grads[b,:,(h*self.stride):(h*self.stride+self.kernel_h),(w*self.stride):(w*self.stride+self.kernel_w)] += (in_grads[b,f,h,w]*self.weights[f,:,:,:])
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        # code here
        batch, in_channel, in_height, in_width = inputs.shape
        x_hat = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        x_hat[ :, :,self.pad:in_height+self.pad, self.pad:in_width+self.pad]=inputs
        out_height, out_width= [int((in_height+2*self.pad-self.pool_height)/self.stride)+1, int((in_width+2*self.pad-self.pool_width)/self.stride)+1]
        outputs = np.zeros((batch, in_channel, out_height, out_width))
        assert self.pool_type in ['max', 'avg'], "pool_type is neither max nor avg"
        def pool(x): return np.max(x) if self.pool_type=='max' else np.mean(x)
        for b in range(batch):
            for f in range(in_channel):
                for (w,h) in [(w,h) for w in range(out_width) for h in range(out_height)]:
                    receiptive_field=x_hat[b,f,(h*self.stride):(h*self.stride+self.pool_height),(w*self.stride):(w*self.stride+self.pool_width)]
                    outputs[b,f,h,w] =pool(receiptive_field)
        #############################################################
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        batch, in_channel, in_height, in_width = inputs.shape
        batch, out_channel, out_height, out_width = in_grads.shape
        x_hat = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        x_hat[ :, :,self.pad:in_height+self.pad, self.pad:in_width+self.pad]=inputs
        out_grads = np.zeros((batch, in_channel, in_height, in_width))
        assert self.pool_type in ['max', 'avg'], "pool_type is neither max nor avg"
        def pool_bwd(x): return x==np.max(x) if self.pool_type=='max' else (np.ones_like(x))/(self.pool_height*self.pool_width)
        for b in range(batch):
            for f in range(in_channel):
                for (w,h) in [(w,h) for w in range(out_width) for h in range(out_height)]:
                    receiptive_field = x_hat[b,f,(h*self.stride):(h*self.stride+self.pool_height),(w*self.stride):(w*self.stride+self.pool_width)]
                    out_grads[b,f,(h*self.stride):(h*self.stride+self.pool_height),(w*self.stride):(w*self.stride+self.pool_width)] += (in_grads[b,f,h,w]*pool_bwd(receiptive_field))
        #############################################################
        return out_grads

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = None
        #############################################################
        # code here
        if self.training:
            np.random.seed(self.seed)
            self.mask = np.random.choice([0,1], size=(inputs.shape), p=[self.ratio, 1-self.ratio])
            outputs = self.mask*inputs/(1-self.ratio)
        else:
            outputs = inputs
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        #############################################################
        # code here
        if self.training:
            out_grads = self.mask*in_grads
        else:
            out_grads = in_grads
        #############################################################
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
        
