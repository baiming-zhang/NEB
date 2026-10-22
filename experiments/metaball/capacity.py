"""Original componentwise-Dirichlet Velocity capacity family."""
from pathlib import Path
import argparse,sys
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--K',type=int,choices=[32,128,256,512,1024],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();import train
    sys.argv=[str(Path(train.__file__)),'--method','Velocity','--K',str(a.K),'--config',str(Path(__file__).parent/f'configs/velocity_capacity_k{a.K}.json'),'--output',a.output]
    train.main()
if __name__=='__main__':main()
