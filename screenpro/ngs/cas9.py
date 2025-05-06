from time import time
import pandas as pd
import numpy as np
import polars as pl
import biobear as bb


def fastq_to_count_single_guide(
        fastq_file_path:str, 
        trim5p_start:int=None, trim5p_length:int=None, 
        verbose: bool=False) -> pl.DataFrame:
    """
    Count the occurrences of unique sequences in single-end FASTQ files to a DataFrame containing counts of unique sequences.
    e.g. single-guide design R1: protospacer

    Args:
        fastq_file_path (str): The path to the FASTQ file.
        trim5p_start (int, optional): The starting position for trimming the 5' end of the sequences. Defaults to None.
        trim5p_length (int, optional): The length of the trimmed sequences. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        pl.DataFrame: A DataFrame containing the unique sequences and their respective counts.
    """
    
    if verbose: ('count unique sequences ...')
    t0 = time()
    
    session = bb.connect()
    
    if trim5p_start and trim5p_length:
        sql_cmd = f"""
        SELECT substr(f.sequence, {trim5p_start}, {trim5p_length}) AS protospacer, COUNT(*) as count
        FROM fastq_scan('{fastq_file_path}') f
        GROUP BY protospacer
        """
    else:
        sql_cmd = f"""
        SELECT f.sequence AS protospacer, COUNT(*) as count
        FROM fastq_scan('{fastq_file_path}') f
        GROUP BY protospacer
        """
    
    df_count = session.sql(sql_cmd).to_polars()

    if verbose: print("done in %0.3fs" % (time() - t0))

    return df_count


def fastq_to_count_dual_guide(
        R1_fastq_file_path:str, R2_fastq_file_path:str,
        trim5p_pos1_start:int=None, trim5p_pos1_length:int=None,
        trim5p_pos2_start:int=None, trim5p_pos2_length:int=None,
        verbose: bool=False) -> pl.DataFrame:
    """
    Count the occurrences of unique sequences in paired-end FASTQ files to a DataFrame containing counts of unique pairs of sequences.
    e.g. dual-guide design R1: protospacer_A, R2: protospacer_B
    
    Args:
        R1_fastq_file_path (str): File path of the R1 FASTQ file.
        R2_fastq_file_path (str): File path of the R2 FASTQ file.
        trim5p_pos1_start (int, optional): Start position for trimming the 5' end of the R1 sequences. Defaults to None.
        trim5p_pos1_length (int, optional): Length of the trimmed R1 sequences. Defaults to None.
        trim5p_pos2_start (int, optional): Start position for trimming the 5' end of the R2 sequences. Defaults to None.
        trim5p_pos2_length (int, optional): Length of the trimmed R2 sequences. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    
    Returns:
        pl.DataFrame: DataFrame containing counts of unique sequences with columns 'protospacer_A', 'protospacer_B', and 'count'.
    
    Raises:
        ValueError: If trim5p_pos1_start, trim5p_pos1_length, trim5p_pos2_start, and trim5p_pos2_length are not provided concurrently.
    """
    
    if verbose: ('count unique sequences ...')
    t0 = time()

    session = bb.connect()

    if trim5p_pos1_start and trim5p_pos1_length and trim5p_pos2_start and trim5p_pos2_length:
        sql_cmd = f"""
            WITH pos1 AS (
                SELECT REPLACE(name, '_R1', '') trimmed_name, *
                FROM fastq_scan('{R1_fastq_file_path}')
            ), pos2 AS (
                SELECT REPLACE(name, '_R2', '') trimmed_name, *
                FROM fastq_scan('{R2_fastq_file_path}')
            )
            SELECT substr(pos1.sequence, {trim5p_pos1_start}, {trim5p_pos1_length}) protospacer_A, reverse_complement(substr(pos2.sequence, {trim5p_pos2_start}, {trim5p_pos2_length})) protospacer_B, COUNT(*) count
            FROM pos1
            JOIN pos2
                ON pos1.name = pos2.name
            GROUP BY protospacer_A, protospacer_B
        """
    elif trim5p_pos1_start==None and trim5p_pos1_length==None and trim5p_pos2_start==None and trim5p_pos2_length==None:
        sql_cmd = f"""
            WITH pos1 AS (
                SELECT REPLACE(name, '_R1', '') trimmed_name, *
                FROM fastq_scan('{R1_fastq_file_path}')
            ), pos2 AS (
                SELECT REPLACE(name, '_R2', '') trimmed_name, *
                FROM fastq_scan('{R2_fastq_file_path}')
            )
            SELECT pos1.sequence protospacer_A, reverse_complement(pos2.sequence) protospacer_B, COUNT(*) count
            FROM pos1
            JOIN pos2
                ON pos1.name = pos2.name
            GROUP BY protospacer_A, protospacer_B
        """
    else:
        raise ValueError("trim5p_pos1_start, trim5p_pos1_length, \
                         trim5p_pos2_start, and trim5p_pos2_length \
                         must be provided concurrently!")

    df_count = session.sql(sql_cmd).to_polars()

    if verbose: print("done in %0.3fs" % (time() - t0))

    return df_count


def map_to_library_single_guide(df_count, library, return_type='all', verbose=False):
    """
    Map the counts of unique sequences to a library DataFrame containing sgRNA sequences.
    User can choose to return mapped reads, unmapped reads, or both.

    Args:
        df_count (pandas.DataFrame): The input DataFrame containing counts.
        library (pandas.DataFrame): The library DataFrame to map to.
        return_type (str, optional): The type of result to return. Defaults to 'all'.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.

    Returns:
        dict or pandas.DataFrame: The mapped result based on the return_type parameter.

    Raises:
        ValueError: If the return_type parameter is invalid.
    """
    # get counts for given input
    res = df_count.clone() #cheap deepcopy/clone
    res = res.sort('count', descending=True)

    res = res.with_columns(
        pl.col("protospacer").alias("sequence"),
    )

    res_map = pl.DataFrame(library).join(
        res, on="sequence", how="left"
    )

    if return_type == 'unmapped' or return_type == 'all':
        res_unmap = res.join(
            pl.DataFrame(library), on="sequence", how="anti"
        )

    if verbose:
        print("% mapped reads",
            100 * \
            res_map.to_pandas()['count'].fillna(0).sum() / \
            int(res.select(pl.sum("count")).to_pandas()['count'])
        )
    
    if return_type == 'unmapped':
        return res_unmap
    elif return_type == 'mapped':
        return res_map
    elif return_type == 'all':
        return {'full': res, 'mapped': res_map, 'unmapped': res_unmap}
    else:
        raise ValueError("return_type must be either 'unmapped', 'mapped', or 'all'")

def hamming_distance(seq1, seq2):
    assert len(seq1) == len(seq2), "Sequences must be of the same length"
    return sum(1 for a, b in zip(seq1, seq2) if a != b)


def map_to_library_dual_guide(df_count, library, get_recombinant=False, get_one_mismatch=False, return_type='all', verbose=False):
    """
    Map the counts of unique sequences to a library DataFrame containing dual-guide sgRNA sequences. 
    Optionally, the function can capture recombinant events.
    User can choose to return mapped reads, unmapped reads, recombinant events, or all.

    Args:
        df_count (pandas.DataFrame): The input DataFrame containing the counts.
        library (pandas.DataFrame): The library of sequences to map against.
        get_recombinant (bool, optional): Whether to calculate recombinant events. Defaults to False.
        get_one_mismatch (bool, optional): Whether to include guide sequences with one mismatch to the library. Defaults to False.
        return_type (str, optional): The type of reads to return. Can be 'unmapped', 'mapped', 'recombinant', or 'all'. Defaults to 'all'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        pandas.DataFrame or dict: The mapped reads based on the specified return_type.

    Raises:
        ValueError: If return_type is not one of 'unmapped', 'mapped', 'recombinant', or 'all'.
        ValueError: If get_recombinant is False and return_type is 'recombinant'.

    """
    # get counts for given input
    res = df_count.clone() #cheap deepcopy/clone
    res = res.rename(
        {'protospacer_a':'protospacer_A','protospacer_b':'protospacer_B'}
        )

    if not get_one_mismatch:
        res = res.sort('count', descending=True)
        res = res.with_columns(
            pl.concat_str(
                [
                    pl.col("protospacer_A"),
                    pl.col("protospacer_B")
                ],
                separator=";",
            ).alias("sequence"),
        )

        # map to library
        res_map = pl.DataFrame(library).join(
                res, on="sequence", how="left"
            )

        # get unmapped reads to the library
        if get_recombinant or return_type == 'unmapped' or return_type == 'all':
            res_unmap = res.join(
                pl.DataFrame(library), on="sequence", how="anti"
            )
    else:
        # Hamming distance 1 mapping
        res_mapped_mask = np.zeros(len(res), dtype=bool)
        counts_one_mismatch = []
        for _, _, _, protospacer_A, _, protospacer_B, _ in library.iter_rows():
            count = 0
            for i, (protospacer_A_res, protospacer_B_res, count_res, _) in enumerate(res.iter_rows()):
                if hamming_distance(protospacer_A, protospacer_A_res) <= 1 and hamming_distance(protospacer_B, protospacer_B_res) <= 1:
                    count += count_res
                    res_mapped_mask[i] = True
            counts_one_mismatch.append(count)
        res = pl.DataFrame(library)
        res = res.with_columns(
            pl.Series(name='count', values=counts_one_mismatch)
        )

        res_map = res.filter(res_mapped_mask)
        res_unmap = res.filter(~res_mapped_mask)

    if verbose:
        print("% mapped reads",
            100 * \
            res_map.to_pandas()['count'].fillna(0).sum() / \
            int(res.select(pl.sum("count")).to_pandas()['count'])
        )
    
    if get_recombinant:
        if verbose:
            print("% unmapped reads",
                100 * \
                res_unmap.to_pandas()['count'].fillna(0).sum() / \
                int(res.select(pl.sum("count")).to_pandas()['count'])
            )
        
        if not get_one_mismatch:
            sgRNA_table = pd.concat([
                library.to_pandas()[['sgID_A','protospacer_A']].rename(columns={'sgID_A':'sgID','protospacer_A':'protospacer'}),
                library.to_pandas()[['sgID_B', 'protospacer_B']].rename(columns={'sgID_B':'sgID','protospacer_B':'protospacer'})
            ]).drop_duplicates(keep='first')

            res_unmap_remapped_a = res_unmap.join(
                pl.DataFrame(sgRNA_table.rename(
                    columns={'protospacer':'protospacer_A','sgID':'sgID_A'})[['sgID_A','protospacer_A']]),
                on=["protospacer_A"], how="left"
            )

            res_recomb_events = res_unmap_remapped_a.join(
                pl.DataFrame(sgRNA_table.rename(
                    columns={'protospacer':'protospacer_B','sgID':'sgID_B'})[['sgID_B','protospacer_B']]),
                on=["protospacer_B"], how="left"
            )

        if get_one_mismatch:
            sgID_A_list = []
            sgID_B_list = []

            for protospacer_A_res, protospacer_B_res, count_res, _ in res_unmap.iter_rows():
                sgID_A = None
                sgID_B = None
                for _, row in sgRNA_table.iterrows():
                    if hamming_distance(protospacer_A_res, row['protospacer']) <= 1:
                        sgID_A = row['sgID']
                    if hamming_distance(protospacer_B_res, row['protospacer']) <= 1:
                        sgID_B = row['sgID']

                sgID_A_list.append(sgID_A)
                sgID_B_list.append(sgID_B)

            res_recomb_events = res_unmap.with_columns(
                pl.Series(name='sgID_A', values=sgID_A_list),
                pl.Series(name='sgID_B', values=sgID_B_list)
            )

        if verbose:
            print("% fully remapped recombination events",
                100 * \
                res_recomb_events.drop_nulls().to_pandas()['count'].fillna(0).sum() / \
                int(res.select(pl.sum("count")).to_pandas()['count'])
            )
    
    if return_type == 'unmapped':
        # TODO: add option to return only unmapped reads after mapping recombinant events
        return res_unmap
    elif return_type == 'mapped':
        return res_map
    elif return_type == 'recombinant':
        if get_recombinant:
            return res_recomb_events
        else:
            raise ValueError("get_recombinant must be set to True to calculate recombinant events")
    elif return_type == 'all':
        if get_recombinant:
            return {'full': res, 'mapped': res_map, 'recombinant': res_recomb_events, 'unmapped': res_unmap}
        else:
            return {'full': res, 'mapped': res_map, 'unmapped': res_unmap}
    else:
        raise ValueError("return_type must be either 'unmapped', 'mapped', 'recombinant', or 'all'")
