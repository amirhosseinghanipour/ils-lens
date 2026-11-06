"""Export actual stage tables and plots without inventing missing measurements."""
import argparse
from pathlib import Path
from ilslens.reporting import render,figures

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--output',default='artifacts/qwen-study')
    p.add_argument('--report',default='QWEN_STUDY_RESULTS.md');args=p.parse_args()
    text,found,missing=render(args.runs,args.output)
    images=figures(found,args.output) if found else []
    for image in images:text+=f'\n![{image}]({Path(args.output)/image})\n'
    text+='\nMissing stages: '+(', '.join(missing) if missing else 'none; substantive completeness still requires the research audit and review')+'\n'
    Path(args.report).write_text(text)
    print(f'Exported {len(found)} measured stages; {len(missing)} missing. No scientific completion claim.')

if __name__=='__main__':main()
