import json,os,statistics,sys
root='/workspace/quant_matrix_calib500'
for run in ['y26n_humanshaped_v2','y26n_noe2e_warm50-2','y26s_humanshaped_smallpatch_v1']:
    for sz in (640,416,320):
        for kind in ('fp32tflite_',''):
            raw=f'{root}/{run}/{kind}sz{sz}_spotval/raw'; n=0; confs=[]; cls={}; hi=0; keys=None
            for f in os.listdir(raw):
                d=json.load(open(os.path.join(raw,f)))
                dets=d.get('detections',[])
                for det in dets:
                    if keys is None: keys=list(det.keys())
                    c=det.get('conf', det.get('confidence', det.get('score'))); confs.append(c); n+=1
                    k=det.get('cls', det.get('class', det.get('class_id', det.get('label')))); cls[k]=cls.get(k,0)+1
                    if c is not None and c>=0.25: hi+=1
            print('DETS', run, sz, kind or 'int8', 'n_dets=%d'%n, 'mean_conf=%.4f'%(statistics.mean(confs) if confs else -1), 'n_conf>=0.25=%d'%hi, 'per_class=%s'%cls, 'keys=%s'%keys, flush=True)
print('DIAG_DONE')
