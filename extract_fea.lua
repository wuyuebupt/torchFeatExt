--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

-- USAGE
-- SINGLE FILE MODE
--          th extract-features.lua [MODEL] [FILE] ...
--
-- BATCH MODE
--          th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES] 
--
      

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'


if #arg < 2 then
   io.stderr:write('Usage (Single file mode): th extract-features.lua [MODEL] [FILE] ... \n')
   os.exit(1)
end


-- get the list of files
local list_of_filenames = {}
local batch_size = 1

if not paths.filep(arg[1]) then
    io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
    os.exit(1)
end

function file_exists(file)
        local f = io.open(file, "rb")
        if f then f:close() end
        return f ~= nil
end

function lines_from(file)
        if not file_exists(file) then return {} end
        lines = {}
        for line in io.lines(file) do
                lines[#lines + 1] = line
        end
        return lines
end

local file = arg[2]
local lines = lines_from(file)


-- Load the model
local model = torch.load(arg[1]):cuda()
-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()
-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}
local features = {}
local images = {}
-- for i=1,number_of_files,batch_size do


local matio = require 'matio'
for k,v in pairs(lines) do
    print (k)

    -- preprocess the images for the batch
    local img = image.load(v, 3, 'float')
    img = transform(img)
    local batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the layer before the (removed) fully connected layer
   local output = model:forward(batch:cuda()):squeeze(1):float()

   -- this is necesary because the model outputs different dimension based on size of input
   if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 
   
   local tmp = torch.Tensor(output:size()):copy(output)
   -- table.insert(features, tmp)
   -- table.insert(images, v) 
   features[v] = tmp
end



torch.save('features.t7', features)
-- matio.save('features.mat', {myStruct = {t1=features}})
print('saved features to features.t7')
