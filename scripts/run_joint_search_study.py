"""One terminal command; preserve completed stages and all interrupted files."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE))
from scripts.run_compound_study import archive_interrupted, runner_lock, complete
from portmatch.joint_search_experiment import check, freeze
from portmatch.compound_experiment import write_json


def status(root):
    if not (root/'protocol.json').exists():
        return {'initialized':False}
    f=check(root)
    return {'initialized':True,'folds':{str(i):'complete' if complete(root/f'fold{i}'/'status.json')
        else 'interrupted' if (root/f'fold{i}').exists() else 'not_started' for i in f['planned_folds']}}


def run(root,workers):
    children,guard,printing,stop=set(),threading.Lock(),threading.Lock(),threading.Event()
    executor=None
    def command(label,args):
        with (root/f'{label}.log').open('w',encoding='utf-8') as log:
            with guard:
                if stop.is_set():return 130
                child=subprocess.Popen([sys.executable,'-u','-m','portmatch.joint_search_experiment',*args,'--output',str(root)],
                    cwd=WORKSPACE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                children.add(child)
            try:
                for line in child.stdout:
                    log.write(line);log.flush()
                    with printing:print(f'[{label}] {line.rstrip()}',flush=True)
                return child.wait()
            except BaseException:
                if child.poll() is None:child.terminate()
                child.wait();raise
            finally:
                child.stdout.close()
                with guard:children.discard(child)
    try:
        state=status(root);print(json.dumps(state,indent=2),flush=True)
        pending=[int(f) for f,s in state['folds'].items() if s!='complete']
        archive_interrupted(root,[name for f in pending for name in (f'fold{f}',f'fold{f}.log')])
        executor=ThreadPoolExecutor(max_workers=workers)
        jobs={executor.submit(command,f'fold{f}',['run','--fold',str(f)]):f for f in pending}
        failed=[]
        for job in as_completed(jobs):
            if job.result():failed.append(jobs[job])
            write_json(root/'orchestration.json',{'failed_folds':failed,'state':status(root)})
        if failed:
            print(f'Failed folds: {failed}; inspect preserved logs.',flush=True);return 1
        return command('report',['report'])
    except KeyboardInterrupt:
        stop.set()
        with guard:
            for child in children:
                if child.poll() is None:child.terminate()
        print('Stopped. Completed folds reusable; interrupted files preserved.',flush=True)
        return 130
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('runs/compound_v1_20261030'))
    p.add_argument('--output',type=Path,default=Path('runs/joint_search_v1_20261031'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--status',action='store_true')
    args=p.parse_args();root=args.output.resolve()
    if args.status:
        print(json.dumps(status(root),indent=2));return 0
    if not root.exists():
        freeze(root,args.parent)
    else:
        f=check(root)
        if Path(f['parent'])!=args.parent.resolve():
            raise ValueError('Requested parent differs from frozen parent')
    with runner_lock(root):return run(root,args.workers)


if __name__=='__main__':
    sys.exit(main())
