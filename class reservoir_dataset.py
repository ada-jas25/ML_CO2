class reservoir_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # let's just store what we input
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # should return the number of files in the directory
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        # need to be able to do this as list
        input_list = []
        output_list = []

        if isinstance(index, int):
            index = [index]

        for idx in index:
            # Convert idx to marker for file
            idx = str(idx)
            L = len(idx)
            idx = idx.rjust(4,"0")
            # That should work

            # Now we load in the file
            with np.load(self.root_dir+"data_"+idx+".npz") as data:
                for key, value in data.items():
                    globals()[key] = value

            # Now we are going to create a numpy array with the data
            input_array = np.zeros((10,96,200))
            nz = porosity.shape[0]
            input_array[0,:nz,:] = porosity
            input_array[1,:nz,:] = perm_r
            input_array[2,:nz,:] = perm_z
            input_array[3,:,:] = np.ones((96,200)) * inj_rate
            input_array[4,:,:] = np.ones((96,200)) * temperature
            input_array[5,:,:] = np.ones((96,200)) * depth
            input_array[6,:,:] = np.ones((96,200)) * Swi
            input_array[7,:,:] = np.ones((96,200)) * lam
            input_array[8,:,:] = np.ones((96,200)) * perf_interval[0]
            input_array[9,:,:] = np.ones((96,200)) * perf_interval[1]

            input_tensor = torch.from_numpy(input_array)

            output_array = np.zeros((2,96,200,24))
            output_array[0,:nz,:,:] = pressure_buildup
            output_array[1,:nz,:,:] = gas_saturation # does this work?

            output_tensor = torch.from_numpy(output_array)

            input_list.append(input_tensor)
            output_list.append(output_tensor)
            #
        sample = torch.stack(input_list,dim=0),torch.stack(output_list,dim=0)

        if self.transform:
            sample = self.transform(sample)

        return sample

# Okay, i think this should work