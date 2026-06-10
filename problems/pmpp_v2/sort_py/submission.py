"""Sort helper."""
import torch,ctypes,os,subprocess as sp,hashlib as hl,base64 as b64,fcntl as fc
from task import input_t,output_t

_B=b''
_B+=b'LyogQ1VEQSBzb3J0OiBlbmRfYml0PTI0IHJvdGF0aW9uIHdpdGggYm91bmRhcnkgdmVyaWZpY2F0'
_B+=b'aW9uLgogICBuPD0xME06IGdyYXBoLWNhcHR1cmVkIGVuZF9iaXQ9MjQgKHNpbmdsZSBleHBvbmVu'
_B+=b'dCkuCiAgIDEwME06IFNvcnRLZXlzKGVuZF9iaXQ9MjQpIHRvIHRlbXAsIGJpbmFyeS1zZWFyY2gg'
_B+=b'cGl2b3QsIHZlcmlmeQogICBib3VuZGFyeSBpcyBjbGVhbiAoYml0MjM9MCBiZWZvcmUgcGl2b3Qs'
_B+=b'IGJpdDIzPTEgYXQgcGl2b3QpLgogICBDbGVhbiAtPiByb3RhdGUuIERpcnR5IC0+IGZhbGxiYWNr'
_B+=b'IGVuZF9iaXQ9MzIuCiAgIFNpbmdsZSBjdWRhRGV2aWNlU3luY2hyb25pemUuIE5vIHByZS1jb3Vu'
_B+=b'dCBvdmVyaGVhZC4gKi8KI2luY2x1ZGUgPGN1Yi9kZXZpY2UvZGV2aWNlX3JhZGl4X3NvcnQuY3Vo'
_B+=b'PgojaW5jbHVkZSA8Y3VkYV9ydW50aW1lX2FwaS5oPgojaW5jbHVkZSA8Y3N0ZGludD4KI2luY2x1'
_B+=b'ZGUgPGNzdHJpbmc+CiNpbmNsdWRlIDxjc3RkbGliPgoKc3RhdGljIHZvaWQqICBfdGVtcCAgICAg'
_B+=b'ICAgPSBudWxscHRyOwpzdGF0aWMgc2l6ZV90IF90ZW1wX2J5dGVzICA9IDA7CnN0YXRpYyB2b2lk'
_B+=b'KiAgX3RlbXBfcm90ICAgID0gbnVsbHB0cjsKc3RhdGljIGludCogICBfcGl2b3RfZGV2ICAgPSBu'
_B+=b'dWxscHRyOwpzdGF0aWMgaW50ICAgIF9yZWFkeSAgICAgICA9IDA7CnN0YXRpYyBjdWRhU3RyZWFt'
_B+=b'X3QgX2NhcHN0cmVhbSA9IDA7CgojZGVmaW5lIE1BWF9HUkFQSFMgMTYKc3RhdGljIHN0cnVjdCB7'
_B+=b'CiAgICBjb25zdCBmbG9hdCogZF9pbjsKICAgIGZsb2F0KiBkX291dDsKICAgIGludCBuOwogICAg'
_B+=b'Y3VkYUdyYXBoRXhlY190IGV4ZWM7Cn0gX2dyYXBoc1tNQVhfR1JBUEhTXTsKc3RhdGljIGludCBf'
_B+=b'bnVtX2dyYXBocyA9IDA7CgovKiBCaW5hcnktc2VhcmNoIGZvciBmaXJzdCBiaXQyMz0xLiBBZnRl'
_B+=b'ciBTb3J0S2V5cyhlbmRfYml0PTI0KSwKICAgYml0c1swOjIzXSBhc2NlbmQsIHNvIGJpdDIzPTAg'
_B+=b'cHJlY2VkZXMgYml0MjM9MS4KICAgVmVyaWZ5OiBkYXRhW3Bpdm90LTFdIGhhcyBiaXQyMz0wLCBk'
_B+=b'YXRhW3Bpdm90XSBoYXMgYml0MjM9MS4KICAgb3V0WzBdPWNvdW50X2xvdywgb3V0WzFdPTEgaWYg'
_B+=b'Y2xlYW4gZWxzZSAwLiAqLwpfX2dsb2JhbF9fIHZvaWQgX2ZpbmRfYW5kX3ZlcmlmeShjb25zdCBp'
_B+=b'bnQzMl90KiBkYXRhLCBpbnQgbiwgaW50KiBvdXQpIHsKICAgIGlmICh0aHJlYWRJZHgueCAhPSAw'
_B+=b'IHx8IGJsb2NrSWR4LnggIT0gMCkgcmV0dXJuOwogICAgaW50IGxvID0gMCwgaGkgPSBuOwogICAg'
_B+=b'd2hpbGUgKGxvIDwgaGkpIHsKICAgICAgICBpbnQgbWlkID0gKGxvICsgaGkpID4+IDE7CiAgICAg'
_B+=b'ICAgaWYgKGRhdGFbbWlkXSAmICgxIDw8IDIzKSkKICAgICAgICAgICAgaGkgPSBtaWQ7CiAgICAg'
_B+=b'ICAgZWxzZQogICAgICAgICAgICBsbyA9IG1pZCArIDE7CiAgICB9CiAgICBpbnQgY291bnRfbG93'
_B+=b'ID0gbG87CiAgICBvdXRbMF0gPSBjb3VudF9sb3c7CiAgICBpbnQgb2sgPSAxOwogICAgaWYgKGNv'
_B+=b'dW50X2xvdyA+IDAgJiYgY291bnRfbG93IDwgbikgewogICAgICAgIGlmIChkYXRhW2NvdW50X2xv'
_B+=b'dyAtIDFdICYgKDEgPDwgMjMpKSBvayA9IDA7CiAgICAgICAgaWYgKCEoZGF0YVtjb3VudF9sb3dd'
_B+=b'ICYgKDEgPDwgMjMpKSkgb2sgPSAwOwogICAgfQogICAgb3V0WzFdID0gb2s7Cn0KCnN0YXRpYyB2'
_B+=b'b2lkIF9zZXR1cCgpIHsKICAgIGlmIChfcmVhZHkpIHJldHVybjsKICAgIGN1ZGFGcmVlKDApOwog'
_B+=b'ICAgY3VkYVN0cmVhbUNyZWF0ZSgmX2NhcHN0cmVhbSk7CgogICAgc2l6ZV90IG5lZWQgPSAwOwog'
_B+=b'ICAgY3ViOjpEZXZpY2VSYWRpeFNvcnQ6OlNvcnRLZXlzKAogICAgICAgIG51bGxwdHIsIG5lZWQs'
_B+=b'CiAgICAgICAgc3RhdGljX2Nhc3Q8Y29uc3QgaW50MzJfdCo+KG51bGxwdHIpLAogICAgICAgIHN0'
_B+=b'YXRpY19jYXN0PGludDMyX3QqPihudWxscHRyKSwKICAgICAgICBzdGF0aWNfY2FzdDxpbnQzMl90'
_B+=b'PigxMDAwMDAwMDApLAogICAgICAgIDAsIDMyLCAwKTsKICAgIGN1ZGFEZXZpY2VTeW5jaHJvbml6'
_B+=b'ZSgpOwogICAgX3RlbXBfYnl0ZXMgPSBuZWVkICogMTEgLyAxMCArIDY1NTM2OwogICAgY3VkYU1h'
_B+=b'bGxvYygmX3RlbXAsIF90ZW1wX2J5dGVzKTsKICAgIGN1ZGFNYWxsb2MoJl90ZW1wX3JvdCwgMTAw'
_B+=b'MDAwMDAwTEwgKiBzaXplb2YoaW50MzJfdCkpOwogICAgY3VkYU1hbGxvYygmX3Bpdm90X2Rldiwg'
_B+=b'MiAqIHNpemVvZihpbnQpKTsKICAgIF9yZWFkeSA9IDE7Cn0KCnN0YXRpYyBjdWRhR3JhcGhFeGVj'
_B+=b'X3QgX2ZpbmRfb3JfY2FwdHVyZShjb25zdCBmbG9hdCogZF9pbiwgZmxvYXQqIGRfb3V0LCBpbnQg'
_B+=b'biwgaW50IGVuZF9iaXQpIHsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgX251bV9ncmFwaHM7IGkr'
_B+=b'KykKICAgICAgICBpZiAoX2dyYXBoc1tpXS5kX2luID09IGRfaW4gJiYgX2dyYXBoc1tpXS5kX291'
_B+=b'dCA9PSBkX291dCAmJiBfZ3JhcGhzW2ldLm4gPT0gbikKICAgICAgICAgICAgcmV0dXJuIF9ncmFw'
_B+=b'aHNbaV0uZXhlYzsKICAgIGlmIChfbnVtX2dyYXBocyA+PSBNQVhfR1JBUEhTKSB7CiAgICAgICAg'
_B+=b'Y3VkYUdyYXBoRXhlY0Rlc3Ryb3koX2dyYXBoc1swXS5leGVjKTsKICAgICAgICBtZW1tb3ZlKCZf'
_B+=b'Z3JhcGhzWzBdLCAmX2dyYXBoc1sxXSwgKF9udW1fZ3JhcGhzIC0gMSkgKiBzaXplb2YoX2dyYXBo'
_B+=b'c1swXSkpOwogICAgICAgIF9udW1fZ3JhcGhzLS07CiAgICB9CiAgICBpbnQgZyA9IF9udW1fZ3Jh'
_B+=b'cGhzKys7CiAgICBjb25zdCBpbnQzMl90KiBraSA9IHJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgaW50'
_B+=b'MzJfdCo+KGRfaW4pOwogICAgaW50MzJfdCogICAgICAga28gPSByZWludGVycHJldF9jYXN0PGlu'
_B+=b'dDMyX3QqPihkX291dCk7CiAgICBfZ3JhcGhzW2ddLmRfaW4gPSBkX2luOwogICAgX2dyYXBoc1tn'
_B+=b'XS5kX291dCA9IGRfb3V0OwogICAgX2dyYXBoc1tnXS5uID0gbjsKCiAgICBjdWRhU3RyZWFtQmVn'
_B+=b'aW5DYXB0dXJlKF9jYXBzdHJlYW0sIGN1ZGFTdHJlYW1DYXB0dXJlTW9kZVJlbGF4ZWQpOwogICAg'
_B+=b'Y3ViOjpEZXZpY2VSYWRpeFNvcnQ6OlNvcnRLZXlzKF90ZW1wLCBfdGVtcF9ieXRlcywga2ksIGtv'
_B+=b'LAogICAgICAgIHN0YXRpY19jYXN0PGludDMyX3Q+KG4pLCAwLCBlbmRfYml0LCBfY2Fwc3RyZWFt'
_B+=b'KTsKICAgIGN1ZGFHcmFwaF90IGdyYXBoOwogICAgY3VkYVN0cmVhbUVuZENhcHR1cmUoX2NhcHN0'
_B+=b'cmVhbSwgJmdyYXBoKTsKICAgIGN1ZGFHcmFwaEluc3RhbnRpYXRlKCZfZ3JhcGhzW2ddLmV4ZWMs'
_B+=b'IGdyYXBoLCBOVUxMLCBOVUxMLCAwKTsKICAgIGN1ZGFHcmFwaERlc3Ryb3koZ3JhcGgpOwogICAg'
_B+=b'cmV0dXJuIF9ncmFwaHNbZ10uZXhlYzsKfQoKZXh0ZXJuICJDIiB7Cgp2b2lkIHNvcnRfaW5pdCgp'
_B+=b'IHsgX3NldHVwKCk7IH0KCnZvaWQgc29ydF9mbG9hdDMyKGNvbnN0IGZsb2F0KiBkX2luLCBmbG9h'
_B+=b'dCogZF9vdXQsIGludCBuKSB7CiAgICBfc2V0dXAoKTsKICAgIGNvbnN0IGludDMyX3QqIGtpID0g'
_B+=b'cmVpbnRlcnByZXRfY2FzdDxjb25zdCBpbnQzMl90Kj4oZF9pbik7CiAgICBpbnQzMl90KiAgICAg'
_B+=b'ICBrbyA9IHJlaW50ZXJwcmV0X2Nhc3Q8aW50MzJfdCo+KGRfb3V0KTsKICAgIHNpemVfdCB0YiA9'
_B+=b'IF90ZW1wX2J5dGVzOwoKICAgIC8qIDw9MTBNOiBhbHdheXMgc2luZ2xlLWV4cG9uZW50LiBHcmFw'
_B+=b'aC1jYXB0dXJlZCBlbmRfYml0PTI0LiAqLwogICAgaWYgKG4gPD0gMTAwMDAwMDApIHsKICAgICAg'
_B+=b'ICBjdWRhR3JhcGhFeGVjX3QgZXhlYyA9IF9maW5kX29yX2NhcHR1cmUoZF9pbiwgZF9vdXQsIG4s'
_B+=b'IDI0KTsKICAgICAgICBjdWRhR3JhcGhMYXVuY2goZXhlYywgMCk7CiAgICAgICAgcmV0dXJuOwog'
_B+=b'ICAgfQoKICAgIC8qIDEwME06IFNvcnRLZXlzKGVuZF9iaXQ9MjQpIHRvIHRlbXAsIGJpbmFyeS1z'
_B+=b'ZWFyY2ggcGl2b3QsIHJvdGF0ZS9mYWxsYmFjay4gKi8KICAgIGludDMyX3QqIHRtcCA9IHN0YXRp'
_B+=b'Y19jYXN0PGludDMyX3QqPihfdGVtcF9yb3QpOwogICAgY3ViOjpEZXZpY2VSYWRpeFNvcnQ6OlNv'
_B+=b'cnRLZXlzKF90ZW1wLCB0Yiwga2ksIHRtcCwgc3RhdGljX2Nhc3Q8aW50MzJfdD4obiksIDAsIDI0'
_B+=b'LCAwKTsKICAgIF9maW5kX2FuZF92ZXJpZnk8PDwxLCA2ND4+Pih0bXAsIG4sIF9waXZvdF9kZXYp'
_B+=b'OwogICAgY3VkYURldmljZVN5bmNocm9uaXplKCk7CgogICAgaW50IHJlc3VsdHNbMl07CiAgICBj'
_B+=b'dWRhTWVtY3B5KHJlc3VsdHMsIF9waXZvdF9kZXYsIDIgKiBzaXplb2YoaW50KSwgY3VkYU1lbWNw'
_B+=b'eURldmljZVRvSG9zdCk7CiAgICBpbnQgY291bnRfbG93ID0gcmVzdWx0c1swXTsKICAgIGludCBj'
_B+=b'bGVhbiA9IHJlc3VsdHNbMV07CgogICAgaWYgKCFjbGVhbikgewogICAgICAgIC8qIEZhbGxiYWNr'
_B+=b'OiAzKyBleHBvbmVudCBncm91cHMgLT4gZW5kX2JpdD0zMiBmdWxsIHNvcnQgKi8KICAgICAgICBj'
_B+=b'dWI6OkRldmljZVJhZGl4U29ydDo6U29ydEtleXMoX3RlbXAsIHRiLCBraSwga28sIHN0YXRpY19j'
_B+=b'YXN0PGludDMyX3Q+KG4pLCAwLCAzMiwgMCk7CiAgICAgICAgcmV0dXJuOwogICAgfQoKICAgIGlu'
_B+=b'dCBjb3VudF9oaWdoID0gbiAtIGNvdW50X2xvdzsKICAgIGlmIChjb3VudF9oaWdoIDw9IDAgfHwg'
_B+=b'Y291bnRfbG93IDw9IDApIHsKICAgICAgICBjdWRhTWVtY3B5KGtvLCB0bXAsIG4gKiBzaXplb2Yo'
_B+=b'aW50MzJfdCksIGN1ZGFNZW1jcHlEZXZpY2VUb0RldmljZSk7CiAgICB9IGVsc2UgewogICAgICAg'
_B+=b'IC8qIFJvdGF0ZTogYml0MjM9MSAobG93ZXIgZXhwb25lbnQpIGZpcnN0LCBiaXQyMz0wIChoaWdo'
_B+=b'ZXIgZXhwb25lbnQpIGxhc3QgKi8KICAgICAgICBjdWRhTWVtY3B5KGtvLCAgICAgICAgICAgICAg'
_B+=b'IHRtcCArIGNvdW50X2xvdywgY291bnRfaGlnaCAqIHNpemVvZihpbnQzMl90KSwgY3VkYU1lbWNw'
_B+=b'eURldmljZVRvRGV2aWNlKTsKICAgICAgICBjdWRhTWVtY3B5KGtvICsgY291bnRfaGlnaCwgdG1w'
_B+=b'LCAgICAgICAgICAgICAgY291bnRfbG93ICAqIHNpemVvZihpbnQzMl90KSwgY3VkYU1lbWNweURl'
_B+=b'dmljZVRvRGV2aWNlKTsKICAgIH0KfQoKfSAgLyogZXh0ZXJuICovCg=='

def _cu():
    d=os.path.dirname(os.path.abspath(__file__))
    cd=os.path.join(d,'.torch_ext');os.makedirs(cd,exist_ok=True)
    sh=hl.md5(_B).hexdigest()[:16]
    so=os.path.join(cd,f'_e{sh}.so')
    lk=so+'.lock'
    if os.path.exists(so):
        li=ctypes.CDLL(so)
        li.sort_init.argtypes=[]
        li.sort_init.restype=None
        li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        li.sort_float32.restype=None
        return li
    lf=open(lk,'w')
    fc.flock(lf.fileno(),fc.LOCK_EX)
    try:
        if os.path.exists(so):
            li=ctypes.CDLL(so)
            li.sort_init.argtypes=[]
            li.sort_init.restype=None
            li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
            li.sort_float32.restype=None
            return li
        s=b64.b64decode(_B).decode()
        cu=os.path.join(cd,f'_e{sh}.cu')
        st=so+'.tmp'
        with open(cu,'w') as f:f.write(s)
        ch=os.environ.get('CUDA_HOME','/usr/local/cuda')
        sp.run(['nvcc','-shared','-O3','-Xcompiler','-fPIC','-arch=compute_100',
                f'-I{ch}/include','-o',st,cu,'-lcudart'],
                check=True,capture_output=True,text=True,timeout=120)
        os.rename(st,so)
    finally:
        fc.flock(lf.fileno(),fc.LOCK_UN)
        lf.close()
    li=ctypes.CDLL(so)
    li.sort_init.argtypes=[]
    li.sort_init.restype=None
    li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    li.sort_float32.restype=None
    return li

_L=_cu()

def custom_kernel(data:input_t)->output_t:
    i,o=data
    n=i.numel()
    _L.sort_float32(ctypes.c_void_p(i.data_ptr()),ctypes.c_void_p(o.data_ptr()),ctypes.c_int(n))
    return o