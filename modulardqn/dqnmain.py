from DQN.Scripts.main import main
from DQN.Utils.cliparser import create_parser


def cli_main():
    parser = create_parser()

    main(parser.parse_args())
