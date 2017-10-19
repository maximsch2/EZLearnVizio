
const vizio_data_h5 = h5open("/scratch/grechkin/vizio_data.h5", "r")
const vizio_data = readmmap(vizio_data_h5["data"])
const vizio_h5_samples = read(vizio_data_h5["samples"]);
const h5_samples_dict = Dict([(sample, i) for (i, sample) in enumerate(vizio_h5_samples)]);
const h5_samples_dict_upcase = Dict([(uppercase(sample), i) for (i, sample) in enumerate(vizio_h5_samples)]);


const vizio_captions_h5 = h5open("/scratch/grechkin/vizio_captions.h5", "r")
const vizio_all_captions = read(vizio_captions_h5["captions"]);



function load_vizio_data()
    VecDataProvider(vizio_data, h5_samples_dict), TextDataProvider(vizio_all_captions, vizio_h5_samples, h5_samples_dict_upcase)
end
