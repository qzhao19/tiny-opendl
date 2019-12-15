



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
            raise NameError('Input placeholder is not Input type!')

        if any(map(lambda x: not isinstance(x, Input), self.output_pls)):
            raise NameError('Output placeholder is not Input type!')

    def find_placeholders_layers(self):
        

    





    




