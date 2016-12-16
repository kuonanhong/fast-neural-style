local display = {} -- main display object
local spatial = assert(require('spatial'))
local win = false
local revclasses

local numericcolours = {
   0xffffff, 0x0000ff, 0x00ff00, 0x008000, 0xff8000, 0x00ffff,
   0x008080, 0xff00ff, 0x800080, 0x8000ff, 0x808000,
   0x808080, 0x404040, 0xff0000, 0x800000, 0xffff00, 0x808000
}

local colours = {
   'white', 'blue', 'green', 'darkGreen', 'orange', 'cyan',
   'darkCyan', 'magenta', 'darkMagenta', 'purple', 'brown',
   'gray', 'darkGray', 'red', 'darkRed', 'yellow', 'darkYellow',
}

local function prep_verbose(opt, source, class_names)

   local targets = opt.targets
   local top_n = opt.dv
   local zoom = opt.zoom or 1
   local targets_names = {}
   -- results integration
   local results_matrix = torch.FloatTensor(#class_names, opt.diw):zero()
   local results_mean = torch.FloatTensor(#class_names):fill(-1)
   local fps_avg = torch.FloatTensor(opt.diw):zero()
   local mean_idx = 1
   local update_results_mean = function(results, mean_idx)
      for i=1, #class_names do
         results_matrix[i][mean_idx] = opt.detection and torch.max(results[i]) or torch.mean(results[i])
         results_mean[i] = torch.mean(results_matrix[i])
      end
   end

   if targets ~= nil and targets ~= '' then
      for i, target in ipairs(targets:split(',')) do
         targets_names[i] = target
      end
   end

   -- calculate display 'zoom' and text offsets
   local side = math.min(source.h, source.w)
   local win_w = zoom*side
   local win_h = zoom*side
   if opt.spatial ~= 0 then
      win_w = zoom*source.w
      win_h = zoom*source.h
   end
   if targets ~= nil and targets ~= '' then
      top_n = #targets_names
   elseif top_n > #class_names then
      top_n = #class_names
   end
   local z = (zoom * side) / 512
   local box_w = 180*z
   local box_h = (60*z)+(20*z*top_n)
   local fps_x = 10*z
   local fps_y = 30*z
   local text_x1 = 10*z
   local text_x2 = 110*z
   local text_y = 20*z
   local txtString

   -- create window
   if not win then
      win = qtwidget.newwindow(win_w, win_h, 'Deep Learning view')
   else
      win:resize(win_w, win_h)
   end

   -- Set font size to a visible dimension
   win:setfontsize(20*z)

   display.forward = function(results, img, fps)

      results = results or {}
      mean_idx = mean_idx % opt.diw + 1
      fps_avg[mean_idx] = fps or 0
      update_results_mean(results, mean_idx)
      local targets_prob = torch.FloatTensor(#targets_names):zero()

      if targets ~= nil and targets ~= '' then
         for i, target in ipairs(targets_names) do
            targets_prob[i] = revClasses[target] and results_mean[ revClasses[target] ] or -1
         end
      end

      win:gbegin()
      win:showpage()

      -- displaying current frame
      image.display{image = img, win = win, zoom = zoom}

      txtString = ''
      if targets == nil or targets == '' then
         _, idx = results_mean:sort(true)
         for i = 1, top_n  do
            if results_mean[ idx[i] ] > results_mean[ idx[1] ] / 3 then
               txtString = txtString .. class_names[ idx[i]]
               if opt.loglevel > 1 then
                  txtString = txtString .. string.format('( %.1f%%)', results_mean[ idx[i] ] * 100)
               end
               txtString = txtString .. ' '
            end
         end
      else
         _, idx = targets_prob:sort(true)
         for i = 1, top_n  do
            if targets_prob[ idx[i] ] > targets_prob[ idx[1] ] / 3 then
               txtString = txtString .. targets_names[ idx[i] ]
               if opt.loglevel > 1 then
                  txtString = txtString .. string.format('( %.1f%%)', targets_prob[ idx[i] ] * 100)
               end
               txtString = txtString .. ' '
            end
         end
      end
      if opt.verbose then
         -- verbose list on right:
         for i = 1, top_n  do
            local move_to_y = text_y*(2+i)
            win:moveto(text_x1, move_to_y)
            if targets == nil or targets == '' then
               win:show(string.format('%s ', class_names[ idx[i] ]))
               win:moveto(text_x2, move_to_y)
               win:show(string.format('(%3.0f%s)', results_mean[ idx[i] ] * 100, '%'))
            else
               win:show(string.format('%s ', targets_names[i]))
               win:moveto(text_x2, move_to_y)
               win:show(string.format('(%3.0f%s)', targets_prob[i] * 100, '%'))
            end
         end
      end
      -- report fps
      win:moveto(10*z, win_h-10*z - 30*z)
      win:setcolor(1, 1, 1)
      win:show(string.format('%.2f fps', torch.mean(fps_avg)))

      -- rectangle for text:
      win:rectangle(0, win_h-30*z, win_h, win_h)
      win:setcolor(0, 0, 0, 0.4)
      win:fill()

      -- report textual string:
      win:setcolor(0.3, 1, 0.3)
      win:moveto(10*z, win_h-10*z)
      win:show(txtString)
      win:gend()

   end -- display.forward

   -- set screen grab function
   display.screen = function() return win:image() end

   -- set continue function
   display.continue = function() return win:valid() end

   -- set close function
   display.close = function()
      if win:valid() then
         win:close()
      end
   end

end


local function prep_simple(opt, source, class_names)

   local targets = opt.targets
   local zoom = opt.zoom or 1
   local targets_names = {}
   -- results integration
   local results_matrix = torch.FloatTensor(#class_names, opt.diw):zero()
   local results_mean = torch.FloatTensor(#class_names):fill(-1)
   local mean_idx = 1
   local update_results_mean = function(results, mean_idx)
      for i=1, #class_names do
         results_matrix[i][mean_idx] = opt.detection and torch.max(results[i]) or torch.mean(results[i])
         results_mean[i] = torch.mean(results_matrix[i])
      end
   end

   if targets ~= nil and targets ~= '' then
      for i, target in ipairs(targets:split(',')) do
         targets_names[i] = target
      end
   end

   -- calculate display 'zoom' and text offsets
   local side = math.min(source.h, source.w)
   local z = (zoom * side) / 512
   local box_x = 5*z
   local box_y = 5*z
   local box_w = 100*z
   local box_h = 40*z
   local text_x = 10*z
   local text_y = 30*z

   local win_w = zoom*side
   local win_h = zoom*side

   if opt.spatial ~= 0 then
      win_w = zoom*source.w
      win_h = zoom*source.h
   end

   -- create window
   if not win then
      win = qtwidget.newwindow(win_w, win_h, 'Image Parser')
   else
      win:resize(win_w, win_h)
   end

   -- Set font size to a visible dimension
   win:setfontsize(20*z)


   display.forward = function(results, img)

      results = results or {}
      mean_idx = mean_idx % opt.diw + 1
      update_results_mean(results, mean_idx)

      win:gbegin()
      win:showpage()

      -- displaying current frame
      image.display{image = img, win = win, zoom = zoom}

      -- rectangle for text:
      win:rectangle(box_x, box_y, box_w, box_h)
      win:setcolor(0, 0, 0, 0.8)
      win:fill()
      if targets == nil or targets == '' then
         _, idx = results_mean:sort(true)
         if opt.pdt <= results_mean[idx[1]] then
            win:setcolor(.8,.8,.8)
            win:moveto(text_x, text_y)
            win:show(string.format('%s ', class_names[idx[1]]))
         end
      else
         local targets_prob = torch.FloatTensor(#targets_names):zero()
         for i, target in ipairs(targets_names) do
            targets_prob[i] = revClasses[target] and results_mean[ revClasses[target] ] or -1
         end
         _, idx = targets_prob:sort(true)
         if opt.pdt <= (targets_prob[idx[1]]) then
            win:setcolor(.8,.8,.8)
            win:moveto(text_x, text_y)
            win:show(string.format('%s ', targets_names[idx[1]]))
         end
      end
      win:gend()

   end -- display.forward

   -- set screen grab function
   display.screen = function() return win:image() end

   -- set continue function
   display.continue = function() return win:valid() end

   -- set close function
   display.close = function()
      if win:valid() then
         win:close()
      end
   end

end

local function prep_spatial_localize(opt, source, class_names)

   local zoom = opt.zoom or 1
   local side = math.min(source.h, source.w)
   local z = side / 512 -- zoom
   local win_w = zoom*source.w
   local win_h = zoom*source.h
   -- fps integration
   local fps_avg = torch.FloatTensor(opt.diw):zero()
   local mean_idx = 1

   -- offset in display (we process scaled and display)
   local offsd = zoom
   if opt.spatial == 1 then
      offsd = zoom*source.h/opt.is
   end

   if not win then
      win = qtwidget.newwindow(win_w, win_h, 'Image Parser')
   else
      win:resize(win_w, win_h)
   end

   -- Set font size to a visible dimension
   win:setfontsize(zoom*20*z)

   display.forward = function(output, img, fps)

      mean_idx = mean_idx % opt.diw + 1
      fps_avg[mean_idx] = fps or 0
      win:gbegin()
      win:showpage()

      -- display frame
      image.display{image = img, win = win, zoom = zoom}

      -- rectangle for text:
      win:rectangle(0, 0, 80, 35)
      win:setcolor(0, 0, 0, 0.4)
      win:fill()

      -- report fps
      win:moveto(10, 20)
      win:setcolor(1, 0.3, 0.3)
      win:show(string.format('%.2f fps', torch.mean(fps_avg)))

      -- visualising prediction
      win:setlinewidth(zoom*3)
      local dsl = spatial.localize(output, class_names, revClasses)

      for i, d in ipairs(dsl) do -- d = detected object!
         -- print bounding boxes of detections
         win:setcolor(colours[(d.idx-1)%#colours+1])

         -- rectangle
         win:rectangle(offsd*d.x1, offsd*d.y1, offsd*d.dx, offsd*d.dy)
         win:stroke()
         win:moveto(offsd*(d.x1+10*z), offsd*(d.y1+30*z))
         win:show(d.class)
         win:moveto(offsd*(d.x1+10*z), offsd*(d.y1+60*z))
         win:show(string.format('(%3.0f%s)', d.value * 100, '%'))
      end
      win:gend()

   end -- display.forward

   -- set screen grab function
   display.screen = function() return win:image() end

   -- set continue function
   display.continue = function() return win:valid() end

   -- set close function
   display.close = function()
      if win:valid() then
         win:close()
      end
   end

end

local function prep_no_gui(opt, source, class_names)

   local targets = opt.targets
   local targets_names = {}
   local targets_maxprob = {}
   local loop_idx = 0

   -- results integration
   local results_matrix = torch.FloatTensor(#class_names, opt.diw):zero()
   local results_mean = torch.FloatTensor(#class_names):fill(-1)
   local mean_idx = 1
   local update_results_mean = function(results, mean_idx)
      for i=1, #class_names do
         results_matrix[i][mean_idx] = opt.detection and torch.max(results[i]) or torch.mean(results[i])
         results_mean[i] = torch.mean(results_matrix[i])
      end
   end

   local pfc = function(...) print(string.format(...)) end
   if opt.noconsole then
      pfc = function(...) end
   end

   if targets ~= nil and targets ~= '' then
      for i, target in ipairs(targets:split(',')) do
         targets_names[i] = target
         targets_maxprob[i] = -1
      end
   end

   display.forward = function(results, img, fps)

      results = results or {}
      mean_idx = mean_idx % opt.diw + 1
      update_results_mean(results, mean_idx)
      loop_idx = loop_idx + 1
      local name = ''
      local prob

      if targets == nil or targets == '' then
         _, idx = results_mean:sort(true)
         if opt.pdt <= (results_mean[idx[1]]) then
            name = class_names[idx[1]]
            prob = results_mean[idx[1]]
         end
      else
         -- Find the most probable target and check that it's higher of the threshold
         name = ''
         maxprob = 0
         local tmpprob
         for i, target in ipairs(targets_names) do
            tmpprob = revClasses[target] and results_mean[ revClasses[target] ] or -1
            if maxprob < tmpprob then
               maxprob = tmpprob
               name = targets_names[i]
            end
         end
         if maxprob > opt.pdt then
            prob = maxprob
         end
      end

      if prob then
         pfc('Iteration %d, processing max possible fps: %.2f, %s %.2f',
            loop_idx, fps, name, prob)
      else
         pfc('Iteration %d, processing max possible fps: %.2f',
            loop_idx, fps)
      end
   end -- display.forward

   -- set continue function
   display.continue = function() return true end

   -- set close function
   display.close = function() end

end

function prep_fixedcam(opt, source, class_names)

   if opt.livecam then
      colours = numericcolours
   else
      require 'qtwidget'
   end
   colours = custom_colours or colours
   local colours_length = #colours
   for i=1, (#class_names) do
      -- replicate colours for extra categories
      local index = (i % colours_length == 0) and colours_length or i%colours_length
      colours[colours_length+i] = colours[index]
   end
   local classes = class_names
   local targets = opt.targets
   local zoom = opt.zoom
   local eye = opt.is
   local diw = opt.diw
   local threshold = opt.pdt
   local loop_idx = 0
   local mean_idx = 0

   local fps_avg = torch.FloatTensor(diw):zero()
   local side = math.min(source.h, source.w)
   local win_w = zoom * source.w
   local win_h = zoom * source.h
   local cam = nil
   local wzoom, hzoom

   if opt.livecam then
      cam = require 'livecam'
      if opt.fullscreen then
         win_w, win_h = cam.window('Camera detector')
      else
         win_w, win_h = cam.window('Camera detector', 0, 0, win_w, win_h)
      end
      wzoom = win_w / source.w
      hzoom = win_h / source.h
   else
      win = qtwidget.newwindow(win_w, win_h, 'Camera detector')
      win:setfontsize(zoom*20)
   end

   display.forward = function(results, img, fps, objects)
      results = {}

      mean_idx = mean_idx + 1
      if mean_idx > diw then
         mean_idx = 1
      end
      fps_avg[mean_idx] = fps or 0

      if cam then
         cam.clear()
         if opt.loglevel > 1 then
            cam.text(15, 25, string.format('%.2f fps', torch.mean(fps_avg)), 20, 0xffffffff)
         end
      else
         win:gbegin()
         win:showpage()

         -- display frame, face image, tracked image:
         image.display{image = img, win = win, zoom = zoom}

         if opt.loglevel > 1 then
            -- rectangle for text:
            win:rectangle(0, 0, zoom*100, 35)
            win:setcolor(0, 0, 0, 0.4)
            win:fill()

            -- report fps
            win:moveto(15, 25)
            win:setcolor('white')
            win:show(string.format('%.2f fps', torch.mean(fps_avg)))
         end

      end
      -- visualizing predictions
      for i, o in ipairs(objects) do
         local d = o.coor

         if o.faceImg and not cam then
            image.display{image = o.faceImg, win = win, zoom = zoom, x=win_w-o.faceImg:size(3)-10, y=zoom*10}
            win:setcolor('green')
            win:setlinewidth(zoom*4)
            win:rectangle(zoom*(win_w-o.faceImg:size(3)-10), zoom*10, zoom*o.faceImg:size(3), zoom*o.faceImg:size(2))
            win:stroke()
         end
         -- if o.trackImg then
            -- win:moveto(source.w-o.trackImg:size(3), source.h-o.trackImg:size(2))
            -- image.display{image = o.trackImg, win = win, zoom = zoom}
         -- end

         if o.target then
            -- plot blobs of target:
            -- outer rectangle
            if cam then
               if not o.colour then
                  o.colour = colours[math.random(#colours)]
               end
               cam.rectangle(wzoom*d.x1, hzoom*d.y1, wzoom*d.w, hzoom*d.h, zoom*8, o.colour)
               -- text:
               if o.faceid then
                  cam.text(wzoom*(d.x1+10), hzoom*(d.y1+30), o.faceid, 20, o.colour)
                  cam.text(wzoom*(win_w-o.faceImg:size(3)), hzoom*30, o.faceid20, o.colour)
               else
                  local text = classes[o.class]
                  if opt.loglevel > 1 then text = text..' '..o.id[1] end
                  cam.text(wzoom*(d.x1+10), hzoom*(d.y1+10), text, 20, o.colour)
               end
            else
               if not o.colour then
                  o.colour = colours[math.random(#colours)]
               end
               win:setlinewidth(zoom*8)
               win:setcolor(o.colour)
               win:rectangle(zoom*d.x1, zoom*d.y1, zoom*d.w, zoom*d.h)
               win:stroke()
               -- text:
               win:moveto(zoom*(d.x1+10), zoom*(d.y1+30))
               if o.faceid then
                  win:show(o.faceid)
                  win:moveto(zoom*(win_w-o.faceImg:size(3)), zoom*30)
                  win:show(o.faceid)
               else
                  local text = classes[o.class]
                  if opt.loglevel > 1 then text = text..' '..o.id[1] end
                  win:show(text)
               end
            end
         else
            -- show as thin gray rectangle:

            if cam then
               cam.rectangle(wzoom*d.x1, hzoom*d.y1, wzoom*d.w, hzoom*d.h, zoom*2, 0xff808080)
            else
               win:setlinewidth(zoom*2)
               win:setcolor('gray')
               win:rectangle(zoom*d.x1, zoom*d.y1, zoom*d.w, zoom*d.h)
               win:stroke()
            end
         end

      end
      if not opt.livecam then
         win:gend()
      end
   end

      -- set screen grab function
   display.screen = function()
      return win and win:image() or nil
   end

   -- set continue function
   display.continue = function()
      return win and win:valid() or true
   end

   -- set close function
   display.close = function()
      if win and win:valid() then
         win:close()
      end
   end

end

--[[

   opt fields
      is          eye size
      spatial     spatial mode 0, 1 or 2
      detection   if we need detection instead of classification
      pdt         threshold for localization
      targets     targets to localize
      zoom        zoom level (defaults to 1)
      diw         number of iterations for averaging
      noconsole   don't print output to console
      gui         gui mode
      dv          number of outputs in verbose gui mode
      localize    localization in gui mode
      loglevel    if > 1, print probabilities in verbose gui mode

   source fields
      w           image width
      h           image height

--]]
function display:init(opt, source, class_names)

   assert(opt and ('table' == type(opt)))
   assert(source and ('table' == type(source)))
   assert(class_names and ('table' == type(class_names)))

   revClasses = {}
   for i,val in ipairs(class_names) do
      revClasses[val] = i
   end

   if opt.fixedcam then
      prep_fixedcam(opt, source, class_names)
   elseif not opt.gui then
      prep_no_gui(opt, source, class_names)
   elseif opt.spatial ~= 0 and opt.localize then
      require 'qtwidget'
      prep_spatial_localize(opt, source, class_names)
   else
      require 'qtwidget'
      if opt.dv > 0 then
         prep_verbose(opt, source, class_names)
      else
         prep_simple(opt, source, class_names)
      end
   end

end

return display
