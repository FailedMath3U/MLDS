"""Exhaustive shuttle boundary audit and signed-image edge cases.
Run with numpy installed: python validate_shuttle_and_images.py
Writes validation/shuttle_and_images.json; original files remain read-only.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mlds_assignment1_shuttle'))
sys.path.insert(0,str(ROOT.parent/'mlds_assignment1_pong'))
from shuttle import Model
from pong_env import preprocess


def main():
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    protected=[ROOT.parent/'mlds_assignment1_shuttle/shuttle.py',
               ROOT.parent/'mlds_assignment1_shuttle/results/shuttle_results.npz']
    before={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    model=Model();states,remaining,costs,nxt=model.arrays();trials=0;max_row_error=0.
    for s in states:
        for a in (0,1):
            left=int(s)-min(15*a,int(s));assert remaining[s,a]==left
            assert costs[s,a]==100*a+2*left
            counts={}
            for j,arrival in enumerate((1,2,3,4,5)):
                target=min(left+arrival,200)
                assert nxt[s,a,j]==target and 0<=target<=200
                counts[target]=counts.get(target,0)+1;trials+=1
            mass=sum(count/5 for count in counts.values())
            max_row_error=max(max_row_error,abs(mass-1));assert abs(mass-1)<1e-12
    assert np.all(remaining[:16,1]==0) and remaining[16,1]==1
    assert np.all(nxt[200,0]==200)
    assert np.array_equal(nxt[200,1],[186,187,188,189,190])
    old=np.zeros((210,160,3),dtype=np.uint8);new=old.copy()
    old[55,40,0]=255;new[55,42,0]=255
    # A stationary paddle block is present in both frames.
    old[75:91,140:144,0]=255;new[75:91,140:144,0]=255
    old_copy=old.copy();new_copy=new.copy()
    difference=preprocess(new)-preprocess(old)
    assert difference[10,20]==-1 and difference[10,21]==1
    assert not np.any(difference[20:28,70:72])
    assert np.array_equal(old,old_copy) and np.array_equal(new,new_copy)
    after={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after
    result={'passed':True,'shuttle':{'state_action_arrival_trials':trials,
            'transition_rows':len(states)*2,'maximum_probability_row_error':max_row_error,
            'underfilled_dispatch_clears_queue':True,'full_queue_wait_probability_at_200':1.,
            'full_queue_dispatch_next':[186,187,188,189,190]},
            'image_difference':{'disappearing_pixel':int(difference[10,20]),
            'appearing_pixel':int(difference[10,21]),'stationary_paddle_difference_is_zero':True,
            'source_images_unchanged':True},'original_files_unchanged':True,'source_sha256':before}
    (out/'shuttle_and_images.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
