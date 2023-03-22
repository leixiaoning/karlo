import os

if False:
    input_dir = ['outputs_test_meiyan_1',
                'outputs_test_meiyan_2',
                'outputs_test_meiyan_3',
                'outputs_test_meiyan_4',
                'outputs_test_meiyan_5',
                'outputs_test_meiyan_6',
                'outputs_test_meiyan_7',
                'outputs_test_meiyan_8',
                'outputs_test_meiyan_9',]
    for i in input_dir:
        i2 = os.listdir(i)
        for j in range(len(i2)):
            j2 = os.path.join(i, i2[j])
            _c = 'cp ' + j2 + ' ' + j2.replace('/', '_lxn_').replace('outputs_test_meiyan', \
                'outputs_test_meiyan/')
            os.system(_c)
    print("done")

if False:
    input_dir = 'outputs_test_meiyan_sr'
    l1 = os.listdir(input_dir)
    for i in range(len(l1)):
        i2 = os.path.join(input_dir, l1[i])



    print("done")


if True:
    #input_dir = 'outputs_test320_1_sr'
    #output_dir = 'outputs_test320_1_sr_select'
    #input_idx = list('1222222122212122222222222123')
    
    #input_dir = 'outputs_test320_2_sr'
    #output_dir = 'outputs_test320_2_sr_select'
    #input_idx = list('32223333222')

    input_dir = 'outputs_test320_3_sr'
    output_dir = 'outputs_test320_3_sr_select'
    input_idx = list('3323')

    os.makedirs(output_dir, exist_ok=True)

    n_style = len(input_idx)
    l1 = os.listdir(input_dir)
    n_img = len(l1)
    n_test = int(n_img/n_style/3)
    for i in range(n_test):
        _start = i*n_style
        for j in range(len(input_idx)):
            
            selected_name = '{}_{}.jpg'.format(_start, (int(input_idx[j])-1))
            _cmd = 'mv ' + os.path.join(input_dir, selected_name) + ' ' + os.path.join(output_dir, selected_name)
            os.system(_cmd)
            _start += 1
    print("done")