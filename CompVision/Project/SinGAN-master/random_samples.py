from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize,imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' % (opt.input_name, opt.gen_start_scale))
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' % (opt.input_name, opt.scale_h, opt.scale_v))
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        if opt.mode == 'random_samples':
            print(opt)
            print("*******************************")
            real = functions.read_image(opt)
            print("real shape", real.shape)
            print("*******************************")
#            real = img.imread("D:\MVA\CompVision\Project\SinGAN-master\Input\Images/Salt_and_Pepper_Golden_Bridge_by_night.jpg")
#            real = real[:,:,:,None]
#            real = real.transpose((3,2,0,1))/255
#            real = torch.from_numpy(real).type(torch.FloatTensor)
#            real = ((real - 0.5)*2).clamp(-1,1)
#            real = real[:,0:3,:,:]
#            print(real.shape)
#            real = imresize_to_shape(real, (735,1024),opt)
#            print(real.shape)
 
            
            functions.adjust_scales2image(real, opt)
#            real = functions.adjust_scales2image(real, opt)
            
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            print("*******************************")
            print("reals shape")
            for i in range(len(reals)):
                print(reals[i].shape)
#            reals = functions.creat_reals_pyramid(real,[],opt)
            print("*******************************")
            in_s = functions.generate_in2coarsest(reals,1,1,opt)
            print("in_s shape after using in2coarsest",in_s.shape)
            print("*******************************")
            print(opt)
            print("*******************************")
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale,num_samples=1,
#                            in_s = in_s
                            )

        elif opt.mode == 'random_samples_arbitrary_sizes':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)





