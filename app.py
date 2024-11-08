import sys
import pydantic_argparse

if __name__ == '__main__':
    from configs.arguments import TrainingArguments
    from configs.STGCN_configs import STGCNConfig
    from configs.GraphWavenet_configs import GraphWavenetConfig
    from configs.STSSL_configs import STSSLConfig
    from configs.MSDR_configs import MSDRConfig
    from configs.MTGNN_configs import MTGNNConfig
    from main import main

    # get st-encoder
    st_encoder = "STGCN"
    if "--st-encoder" in sys.argv:
        st_encoder = sys.argv[sys.argv.index("--st-encoder") + 1]

    def parse_args(arguments):
        parser = pydantic_argparse.ArgumentParser(
            model=arguments,
            prog="python app.py",
            description="Training model job.",
            version="0.0.1",
            epilog="Training model job.",
        )

        return parser.parse_typed_args()

    args = parse_args(locals()[f"{st_encoder}Config"])
    sys.exit(main(args))
