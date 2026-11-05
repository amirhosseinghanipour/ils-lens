import argparse,json
import yaml

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['toy'])
    parser.add_argument('--config',default='configs/toy.yaml')
    args=parser.parse_args()
    with open(args.config) as file:
        config=yaml.safe_load(file)
    from .eval.toy_hmm import run
    print(json.dumps(run(config),indent=2))

if __name__=='__main__': main()
