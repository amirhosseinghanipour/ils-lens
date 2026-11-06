import argparse,json,sys
import yaml

def main():
    if len(sys.argv)>1 and sys.argv[1]=='study':
        from .study import main as study_main
        study_main(sys.argv[2:]);return
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['toy'],help='Use `ilslens study --help` for guarded serial GPU experiments')
    parser.add_argument('--config',default='configs/toy.yaml')
    args=parser.parse_args()
    with open(args.config) as file:
        config=yaml.safe_load(file)
    from .eval.toy_hmm import run
    print(json.dumps(run(config),indent=2))

if __name__=='__main__': main()
