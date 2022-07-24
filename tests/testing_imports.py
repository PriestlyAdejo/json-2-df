import immediate_tests as im_test

print("\n\nMain encapsulation 1\n\n")
test_dict_1 = dict(flat_td_1=0, dict_td_1=dict(a=1, d=4), nested_td_1=dict(f=dict(r=34, g=44), p=2))
igs_1 = {"cols": ["f"], "name_lvls": [0]}
nskwargs_1 = dict(sep=".", return_dels=True, ignore=igs_1)
n_d, d_t = im_test.main(test_dict_1, nskwargs_1)
