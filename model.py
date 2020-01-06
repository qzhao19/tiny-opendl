from inputs import *
from losses import *
from optimizers import *
from layers import *
from tensors import *
import numpy as np
from .utils.data_ops_utils import *


class Model(object):
    """The `Model` class adds training & evaluation routines to a `Network`.

    Attributs:
        layers: Layer, network layer e.g. cnn, add, reshape, Activation
        input_pls: list, input placeholders
        output_pls: list, output placeholder

    """
    def __init__(self, input_pls, output_pls):
        """init model params 
        """
        self.layers = set()
        self.all_pls = {}

        # input/output placeholder must be a list
        self.input_pls = input_pls if isinstance(input_pls, list) else [input_pls]
        self.output_pls = output_pls if isinstance(output_pls, list) else [output_pls]

        # element of placeholder type should be Input
        if any(map(lambda x: not isinstance(x, Input), self.input_pls)):
            raise TypeError('Input placeholder is not Input type!')

        if any(map(lambda x: not isinstance(x, Input), self.output_pls)):
            raise TypeError('Output placeholder is not Input type!')

    def find_placeholders_layers(self):
        """Iterate over all placeholders and layers from the output layer, 
        save and set the id
        """
        all_pls = {}
        queue = []

        for pl in self.output_pls:
            # backwards from output layer and add into a queue
            id = unique_id('placeholder')
            pl.set_id(id)
            all_pls[id] = pl
            queue.append(pl)

        while len(queue) > 0:
            current_pl = queue[0]
            depend_pls = current_pl.depend_pls

            # get all of layers
            if current_pl.input_layer is not None and current_pl.input_layer is not self.layers:
                self.layers.add(current_pl.input_layer)

            # backward to origin
            if depend_pls is None:
                queue.pop(0)
                continue

            for depend_pl in depend_pls:
                # dont need to process placeholders that have been processed
                if depend_pl.id is not None:
                    continue
                
                # get all of placeholders
                pl_id = unique_id('placeholder')
                depend_pl.set_id(pl_id)
                all_pls[pl_id] = depend_pl
                queue.append(depend_pl)

            # pop a placeholder has been processed and do next
            queue.pop(0)

        self.all_pls = all_pls

    def forward(self, input_tensors):
        """Iterate and calculate all of placeholder and tensor from input layer
        """
        # define 2 set type varaible: ready_pl_ids, waiting_pl_ids
        ready_pl_ids = set()
        waiting_pl_ids = set(self.all_pls.keys())

        # we start to assign value from input layer
        for i, input_pl in enumerate(self.input_pls):
            input_pl.set_tensor(input_tensors[i])
            ready_pl_ids.add(input_pl.id)
            waiting_pl_ids.remove(input_pl.id)
        
        # iterate all placeholder
        while len(waiting_pl_ids) > 0:
            next_pl = None
            for waiting_pl_id in waiting_pl_ids:
                waiting_pl = self.all_pls[waiting_pl_id]
                depend_pls = waiting_pl.depend_pls

            # set all upstream dependent node to ready
            if any(map(lambda x: x.id not in ready_pl_ids, depend_pls)):
                continue
            
            # list depend tensor of current node 
            depend_tensors = [depend_pl.tensor for depend_pl in depend_pls]

             # compute value of current node
            if len(depend_tensors) == 1:
                output_tensor = waiting_pl.input_layer.forward(depend_tensors[0])
            elif len(depend_tensors) == 2:
                output_tensor = waiting_pl.input_layer.forward(depend_tensors[0], depend_tensors[1], )
            else:
                raise('depend tensors numbuer should be 1 or 2!')

            waiting_pl.set_tensor(output_tensor)
            next_pl = waiting_pl
            break
                
        # 
        if next_pl is not None:
            ready_pl_ids.add(waiting_pl.id)
            waiting_pl_ids.remove(waiting_pl.id)
            

    def complie(self, optimizer, loss, validation_data=None):
        """set up model optimizer, loss function and validation dataset
        """
        if not isinstance(optmizer, Optimizer):
            raise TypeError('Input optimizer is not Optimizer type!')

        if not isinstance(loss, Loss):
            raise TypeError('Input loss is not Loss type!')

        self.optimizer = optimizer
        self.loss = loss
        self.find_placeholders_layers()
        self.optimizer.set_layers(self.layers)

    def get_batch_data(self, data):
        return data

    
    def fit(self, input_X, input_y, n_epochs, batch_size=None):
        """training model
        """
        
        
