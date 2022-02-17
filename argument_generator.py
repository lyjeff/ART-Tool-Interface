from argparse import ArgumentParser


class Argument_Generator():
    class __Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

    def generator(self):

        parser = ArgumentParser()

        parser.add_argument('--interface', type=int, default=1,
                            help='Set to select parameters by which interface (1:GUI, 2:Standard Input, 3:only set on arguments) (default: 1)')
        parser.add_argument('--cuda', type=int, default=0,
                            help='set the model to run on which gpu (default: 0)')

        # dataset argument
        parser.add_argument('--dataset-path', type=str,
                            help='training dataset path (default: None)')
        parser.add_argument('--num-workers', type=int, default=8,
                            help='set the number of processes to run (default: 8)')
        parser.add_argument('--norm', action="store_true", default=False,
                            help='set to normalize the dataset (default: False)')

        # model argument
        parser.add_argument(
            '--predict-model', type=int,
            choices=[self.__Range(1, 6)], default=1,
            metavar='PREDICT_MODEL',
            help=f'select the Predict model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101) (default: 1)'
        )
        parser.add_argument(
            '--attack-model', type=int,
            choices=[self.__Range(1, 6)], default=1,
            metavar='ATTACK_MODEL',
            help=f'select attack model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101) (default: 1)'
        )
        parser.add_argument('--white-box', action="store_true", default=False,
                            help='set to attack by white box (default: False)')

        # attack argument
        parser.add_argument(
            '--attack-func', type=int,
            choices=[self.__Range(1, 5)], default=1,
            metavar='ATTACK_FUNCTION',
            help=f'select Attack function (1:FGSM, 2:BIM, 3:PGD, 4:C&W L2, 5:C&W Linf) (default: 1)'
        )
        parser.add_argument('--max-iter', type=int, default=20,
                            help=f'set max_iter  (default: 20)')
        parser.add_argument(
            '--eps', type=float,
            choices=[self.__Range(0.0, 1.0)], default=0.1,
            metavar='EPS',
            help=f'set eps (0.0 <= eps <= 1.0) (default: 0.1)'
        )
        parser.add_argument(
            '--conf', type=int,
            choices=[self.__Range(0, 16)], default=None,
            metavar='CONFIDENCE',
            help=f'set confidence if choice C&W attack function(0 <= conf <= 16) (default: None)'
        )

        # training argument
        parser.add_argument('--epochs', type=int, default=20,
                            help='set the epochs (default: 20)')
        parser.add_argument('--batch-size', type=int, default=32,
                            help='set the batch size (default: 32)')

        # optimizer argument
        parser.add_argument('--optim', type=str, default='SGD',
                            help='set optimizer (default: SGD)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='set the learning rate (default: 1e-3)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='set momentum of SGD (default: 0.9)')

        args = parser.parse_args()

        return args
