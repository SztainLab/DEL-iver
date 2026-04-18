import DEL_iver as deliv
print("finished importing")




output="./Results"
input="./data/20_k_example.csv"
bb_cols=["buildingblock1_smiles","buildingblock2_smiles","buildingblock3_smiles"]
label="binds"

metric="pbind"
prefix="example"

ddr = deliv.DataReader.from_csv(input,building_blocks=bb_cols,output_path=output,label=label)
table, id_to_smile = deliv.generate_bb_dictionaries(ddr)




#Chemical Analysis
bb_stats, disynthon_stats = deliv.compute_pbind_and_enrichment(ddr,ignore_position=True,min_occurrences=0)

deliv.data_set_statistics(ddr)


#top_bb = deliv.find_best_bb(ddr, 10,min_occurrences=30,sort_by=metric, exclude=[1])

#top_disynthons = deliv.find_best_disynthon(ddr, 10,min_occurrences=1,sort_by=metric,exclude=[(1, 2), (1, 3)])

#deliv.draw_bb(top_bb,ddr,metric="pbind",save_svg_path="bb_structures.svg")   

#deliv.draw_disynthons(top_disynthons,ddr,metric=metric,save_svg_path="disynthons_structures.svg")

#deliv.plot_disynthons(ddr,elev=15,azim=35,min_occurrences=30,output_path=f"disinthons.jpg")

#deliv.plot_bb(ddr,output_path=f"bbs.jpg",exclude_bb1=True)


