import json, re, unicodedata, Levenshtein
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict

current_step_number: int = 0

def begin_next_step(explanation: str, turn_json: bool = False, 
                    json_data: List | Dict | None = None) -> None:
    global current_step_number

    if turn_json is True:
        with open(f"metadata_filter_step{current_step_number}.json", 'w') as f:
            json.dump(json_data, f, indent=4)
        
    current_step_number += 1
    print(f"------- step {current_step_number}: {explanation} -------")

def preprocess_str(orig_str: str, is_title: bool = False) -> str:
    nfkd_form = unicodedata.normalize("NFKD", orig_str)
    revised_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    if is_title is True:
        re_pattern = re.compile(r"[^.;:/\u0020]+")
        matches = re_pattern.findall(revised_str)
        return " ".join(matches)
    else:
        return revised_str

def merge_two_instances(str_01: str, str_02 : str, similarity_flag : bool, 
                        correspondence_dict: Dict[str, int],
                        cur_group_idx: int) -> int:
    
    # case 01: when they are the same instance
    if similarity_flag is True:
        group_01: int = correspondence_dict.get(str_01, None)
        group_02: int = correspondence_dict.get(str_02, None)

        # case 01.01: same instance, both str_01 and str_2 are unassigned
        if group_01 is None and group_02 is None:
            correspondence_dict[str_01] = cur_group_idx
            correspondence_dict[str_02] = cur_group_idx
            cur_group_idx += 1
        
        # case 01.02: same instance, either str_01 or str_2 is unassigned
        elif group_02 is None:
            correspondence_dict[str_02] = group_01
        elif group_01 is None:
            correspondence_dict[str_01] = group_02
        
        # case 01.03: same instance, both are assigned
        else:
            if group_01 != group_02:
                group_num: int = group_01 if group_01 < group_02 else group_02
                changeable_arr: List[str] = []
                for key, value in correspondence_dict.items():
                    if value == group_01 or value == group_02:
                        changeable_arr.append(key)

                for changeable in changeable_arr:
                    correspondence_dict[changeable] = group_num

    # case 02: when they are the different authors  
    else:
        if correspondence_dict.get(str_01, None) is None:
            correspondence_dict[str_01] = cur_group_idx
            cur_group_idx += 1
        if correspondence_dict.get(str_02, None) is None:
            correspondence_dict[str_02] = cur_group_idx
            cur_group_idx += 1
    
    return cur_group_idx

def deduplicate_instances(input_dir, output_dir,
                          author_column='author', title_column='title',
                          debug=False, author_levratio=1.0, title_levratio=1.0, 
                          title_filter_criteria='year', filter_ascending=True):

    # ensure both input_dir and output_dir are processable format
    assert output_dir.endswith('.csv')

    print(f"opening input file at {input_dir}")
    if input_dir.endswith('.csv'):
        df = pd.read_csv(input_dir)
    elif input_dir.endswith('.jsonl.gz'):
        df = pd.read_json(input_dir, lines=True, compression='gzip')
    else:
        raise Exception(f"currently not accepting other file format")

    # ------ STEP 1 ------ 
    begin_next_step(explanation="preprocess / regularize all strings")

    origin_length = len(df)
    df = df[~df[author_column].isna()].reset_index().drop(['index'], axis=1)
    df = df[~df[title_column].isna()].reset_index().drop(['index'], axis=1)
    print(f"the metadata file has an original length of {origin_length}," + 
         f"after filtering out instances with no title or author, remaining {len(df)} instances")

    df["regularized_author"] = df[author_column].map(lambda x: preprocess_str(x, is_title=False))
    df["regularized_title"] = df[title_column].map(lambda x: preprocess_str(x, is_title=True))

    print(f"there are a total of {len(df)} text instances in the metadata")

    # ------ STEP 2 ------ 
    begin_next_step(explanation="try aligning and deduplicating all author names")

    # Here the assumption is that all author names follow the format below: 
    # xxx [last name], xxx (xxx) [some kind of first name]
    # whereas the first name can be in any kind of abbreviated form
    
    unique_author_names: List[str] = df["regularized_author"].unique().tolist()
    print(f"there are {len(unique_author_names)} unique authors")
    
    author_name_grouping_by_abbre: Dict[str, Dict[str, List[int]]] = {}
    updated_author_name_dict : Dict[str, str] = {}

    for author_name in tqdm(unique_author_names):

        original_author_name = author_name
        author_name = re.sub(r"&#xbf;", "", author_name)

        # take out the modifier letter prime for now
        # if all character names are not in latin characters, please be careful with this
        author_name = re.sub(r"[^\u0000-\u007F]", "", author_name)
        author_name = re.sub(r"(?<!\b\w)\b(\w{2,})\.", r"\1", author_name)
        author_name = re.sub(r"([A-Z]\.)([A-Z]\.)", r"\1 \2", author_name)

        if len(author_name) == 0:
            updated_author_name_dict[original_author_name] = author_name
            continue

        if author_name.endswith(","):
            author_name = author_name[:-1]
            
        if author_name.endswith("]"):
            if author_name.startswith("["):
                author_name = author_name[1:-1]
            else:
                author_name = author_name.split("[")[0]

        # get the author name abbreviation
        last_first_name_split: List[str] = author_name.split(", ")
        last_name: str = last_first_name_split[0]
        last_name = re.sub(r"\s+", "", last_name)
        if len(last_name) == 0:
            updated_author_name_dict[original_author_name] = author_name
            continue
        author_name = last_name + ", " + ", ".join(last_first_name_split[1:])
        inits_abbr: str = last_name[0] + ".,"
        updated_author_name_dict[original_author_name] = author_name

        if len(last_first_name_split) > 1:
            other_names: str = ", ".join(author_name.split(", ")[1:])
            other_names: List[str] = other_names.split(" ")

            inits_abbr += (
                    " " + other_names[0][0] + "." if len(other_names[0]) > 0 else ""
                )

        author_name_grouping_by_abbre.setdefault(inits_abbr, [])
        author_name_grouping_by_abbre[inits_abbr].append(author_name)
        
    df['regularized_author'] = df['regularized_author'].map(lambda x : updated_author_name_dict[x])
    print(f"found a total of {len(author_name_grouping_by_abbre)} unique author abbreviations")

    # ------ STEP 3 ------ 
    begin_next_step(explanation="filtering out author names using Levenshtein similarity",
                    turn_json=debug, json_data = author_name_grouping_by_abbre)

    author_name_alignment_01 = {}
    for abbrev, author_arr in tqdm(author_name_grouping_by_abbre.items()):

        author_correspondence_dict = {}
        group_idx = 0

        if len(author_arr) == 1:
            author_correspondence_dict[author_arr[0]] = group_idx
            group_idx += 1

        for i in range(len(author_arr) - 1):
            for j in range(i + 1, len(author_arr)):
                name_01 = author_arr[i]
                name_02 = author_arr[j]

                similarity_flag: bool = False

                last_name_sim_ratio = Levenshtein.ratio(
                    name_01.split(", ")[0], 
                    name_02.split(", ")[0])

                if last_name_sim_ratio > author_levratio:
                    similarity_flag = True
                    other_names_01 = (", ".join(name_01.split(", ")[1:])).split(" ")
                    other_names_01 = [name for name in other_names_01 if name != ""]
                    other_names_02 = (", ".join(name_02.split(", ")[1:])).split(" ")
                    other_names_02 = [name for name in other_names_02 if name != ""]

                    shorter_length =  (len(other_names_01) 
                                       if len(other_names_01) < len(other_names_02) 
                                       else len(other_names_02))

                    for idx in range(shorter_length):
                        other_name_01 = other_names_01[idx]
                        other_name_02 = other_names_02[idx]

                        # abbreviation
                        if ((len(other_name_01) == 2 and other_name_01[-1] == ".") 
                            or (len(other_name_02) == 2 and other_name_01[-1] == ".")):
                            
                            if other_name_01[0] != other_name_02[0]:
                                similarity_flag = False
                                break

                        # non abbreviation
                        else:
                            if (Levenshtein.ratio(
                                other_name_01, 
                                other_name_02) < author_levratio):
                            
                                similarity_flag = False
                                break
                    
                group_idx = merge_two_instances(name_01, name_02, 
                                                similarity_flag, author_correspondence_dict, 
                                                group_idx)    
                        
        author_name_alignment_01[abbrev] = author_correspondence_dict

    # ------ STEP 4 ------ 
    begin_next_step(explanation="organize the previously grouped authors",
                    turn_json=debug, json_data = author_name_alignment_01)

    author_name_alignment_02 = {}
    for _, name_dict in tqdm(author_name_alignment_01.items()):
        reversed_data: Dict[int, Tuple[str, List[str]]] = {}

        for name, group_num in name_dict.items():
            reversed_data.setdefault(group_num, ("", []))

            first_name = (
                name
                if len(reversed_data[group_num][0]) < len(name)
                else reversed_data[group_num][0]
            )
            second_list = reversed_data[group_num][1]
            second_list.append(name)

            reversed_data[group_num] = (first_name, second_list)

        for _, (representative_name, name_arr) in reversed_data.items():
            author_name_alignment_02[representative_name] = name_arr
    
    # ------ STEP 5 ------ 
    begin_next_step(explanation="normalize all author names",
                    turn_json=debug, json_data = author_name_alignment_02)
    
    authorship_column: List[str | None] = [None for _ in range(len(df))]


    for normalized_author, author_list in tqdm(author_name_alignment_02.items()):
        author_filtered_indeces: List[int] = []
        for author_name in author_list:
            author_filtered_indeces.extend(
                df[df["regularized_author"] == author_name].index
            )
        
        authorship_column = [
            normalized_author if idx in author_filtered_indeces else author_name
            for idx, author_name in enumerate(authorship_column)
            ]

    df["normalized_author"] = authorship_column

    print(f"there are {len(df['regularized_author'].unique())} unique authors previously, \n"
        + f"there are {len(df['normalized_author'].unique())} unique authors after normalization")
    
    # ------ STEP 6 ------ 
    begin_next_step(explanation="try aligning and deduplicating all title names")

    title_name_grouping_by_author = {}
    unique_authorship_list: List[str] = df['normalized_author'].unique().tolist()
    
    for author_name in tqdm(unique_authorship_list):
            title_name_grouping_by_author[author_name] = (
                (df[df['normalized_author'] == author_name])["regularized_title"]
                .unique()
                .tolist()
            )
    # ------ STEP 7 ------      
    begin_next_step(explanation="filtering out title names using Levenshtein similarity",
                    turn_json=debug, 
                    json_data=title_name_grouping_by_author)
    
    title_name_alignment_01 = {}
    for author_name, title_arr in tqdm(title_name_grouping_by_author.items()):
        
        correspondence_dict: Dict[str, int] = {}
            
        group_idx = 0

        if len(title_arr) == 1:
            correspondence_dict[title_arr[0]] = group_idx
            group_idx += 1
        for i in range(len(title_arr) - 1):
            for j in range(i + 1, len(title_arr)):
                title_01: str = title_arr[i]
                title_02: str = title_arr[j]

                if len(title_01) < len(title_02):
                    title_02_arr = title_02.split(" ")

                    idx_before = 0
                    shortened_title_02 = ""
                    
                    while len(shortened_title_02) < len(title_01):
                        idx_before += 1
                        shortened_title_02 = " ".join(title_02_arr[:idx_before])
                        shortened_title_01 = title_01
                elif len(title_01) > len(title_02):
                    title_01_arr = title_01.split(" ")

                    idx_before = 0
                    shortened_title_01 = ""
                    while len(shortened_title_01) < len(title_02):
                        idx_before += 1
                        shortened_title_01 = " ".join(title_01_arr[:idx_before])
                        shortened_title_02 = title_02
                else:
                    shortened_title_02 = title_02
                    shortened_title_01 = title_01
                sim_ratio = Levenshtein.ratio(
                        shortened_title_01, 
                        shortened_title_02
                    )
                
                similarity_flag = (sim_ratio > title_levratio)

                group_idx = merge_two_instances(title_01, title_02, similarity_flag,
                                                correspondence_dict, group_idx)
                

        title_name_alignment_01[author_name] = correspondence_dict

    # ------ STEP 8 ------      
    begin_next_step(explanation="organize the previously grouped authors",
                    turn_json=debug, 
                    json_data=title_name_alignment_01)

    title_name_alignment_02: Dict[str, Dict[str, List[str]]] = {}

    for author_name, name_dict in tqdm(title_name_alignment_01.items()):

        reversed_data: Dict[int, Tuple[str, List[str]]] = {}

        for title_name, group_num in name_dict.items():

            reversed_data.setdefault(group_num, ("", []))

            first_name = (
                title_name
                if len(reversed_data[group_num][0]) < len(title_name)
                else reversed_data[group_num][0]
                )
            second_list = reversed_data[group_num][1]
            second_list.append(title_name)

            reversed_data[group_num] = (first_name, second_list)

        all_title_dict: Dict[str, List[str]] = {}

        for _, (representative_name, name_arr) in reversed_data.items():
            all_title_dict[representative_name] = name_arr

        title_name_alignment_02[author_name] = all_title_dict

    # ------ STEP 9 ------      
    begin_next_step(explanation="normalize all title names",
                    turn_json=debug, 
                    json_data=title_name_alignment_02)
    
    normalized_title_column: List[str | None] = [None for _ in range(len(df))]

    for author, title_dict in tqdm(title_name_alignment_02.items()):

        for normalized_title, title_arr in title_dict.items():
            title_filtered_indeces: List[int] = []

            for title_name in title_arr:

                title_filtered_indeces.extend(
                    df[
                        (df["normalized_author"] == author)
                        & (df["regularized_title"] == title_name)
                    ].index
                    )

            normalized_title_column = [
                normalized_title if idx in title_filtered_indeces else title_name
                for idx, title_name in enumerate(normalized_title_column)
                ]

    df["normalized_title"] = normalized_title_column
    df = df[~df['normalized_title'].isna()].reset_index().drop(['index'], axis=1)
    df["complete_work_info"] = df.apply(
            lambda x: x["normalized_title"] + " [-] " + x["normalized_author"], axis=1
        )
    unique_title_plus_author_instances: List[str] = (
        df["complete_work_info"].unique().tolist()
        )
    
    print(
            f"among a total of {len(df)} instances," +
            f"there are {len(unique_title_plus_author_instances)} unique title and author"
        )

    print(f"filtering based on column {title_filter_criteria} in " + 
          f"{'ascending' if filter_ascending is True else 'descending'} order")

    filtered_indices: List[int] = []
    for unique_instance in tqdm(unique_title_plus_author_instances):
        filtered_indices.append(
            df[df["complete_work_info"] == unique_instance]
                .sort_values(by=title_filter_criteria, ascending=filter_ascending)
                .index[0]
            )

    filtered_indices = list(set(filtered_indices))
    df = df.iloc[filtered_indices].reset_index().drop(['index'], axis=1)
    
    print(f"complete filtering. writing to {output_dir}")
    df.to_csv(output_dir)