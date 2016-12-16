require 'torch'
require 'nn'
require 'image'
require 'camera'
local video = require 'libvideo_decoder'

require 'qt'
require 'qttorch'
require 'qtwidget'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-models', 'models/instance_norm/mosaic.t7')
cmd:option('-secondspermodel', 600)
cmd:option('-videopath', '')
cmd:option('-height', 800)
cmd:option('-width',  600)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)


local function main()
  local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local models = {}

  local preprocess_method = nil
  local num_models = 0
  for _, checkpoint_path in ipairs(opt.models:split(',')) do
    print('loading model from ', checkpoint_path)
    local checkpoint = torch.load(checkpoint_path)
    local model = checkpoint.model
    model:evaluate()
    model:type(dtype)
    if use_cudnn then
      cudnn.convert(model, cudnn)
    end
    table.insert(models, model)
    num_models = num_models + 1
    local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    if not preprocess_method then
      print('got here')
      preprocess_method = this_preprocess_method
      print(preprocess_method)
    else
      if this_preprocess_method ~= preprocess_method then
        error('All models must use the same preprocessing')
      end
    end
  end

  local preprocess = preprocess[preprocess_method]

  print("starting video", opt.videopath)
  local status, height, width, length, fps = video.init(opt.videopath)
  video.startremux('dummyfilename', 'mp4', 0)
  print(status, height, width, length, fps)

  local win = nil
  local img_raw = torch.ByteTensor(3, height, width)
  local H = opt.height
  local W = opt.width
  while true do
    -- Grab a frame from the webcam
    video.frame_rgb(img_raw)

    -- Preprocess the frame
    --local H, W = img_raw:size(2), img_raw:size(3)
    local img_pre = preprocess.preprocess(image.scale(img_raw, H, W):view(1, 3, W, H)):type(dtype)

    -- Run the models
    local modelid = math.floor(os.time() / opt.secondspermodel ) % num_models
    local img_out_pre = models[modelid+1]:forward(img_pre)
    -- Deprocess the frame and show the image
    local img_out = preprocess.deprocess(img_out_pre)[1]:float()

    --for i, model in ipairs(models) do
      --local img_out_pre = model:forward(img_pre)

      ---- Deprocess the frame and show the image
      --local img_out = preprocess.deprocess(img_out_pre)[1]:float()
      --table.insert(imgs_out, img_out)
    --end
    --
    local img_disp = image.toDisplayTensor{
      input = img_out,
      min = 0,
      max = 1,
      nrow = 1,
    }


    if not win then
      -- On the first call use image.display to construct a window
      win = image.display(img_disp)
    else
      -- Reuse the same window
      win.image = img_out
      local size = win.window.size:totable()
      local qt_img = qt.QImage.fromTensor(img_disp)
      win.painter:image(0, 0, size.width, size.height, qt_img)
    end
  end
end


main()

