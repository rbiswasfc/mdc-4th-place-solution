import pathlib
import re
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from joblib import Parallel, delayed
from Levenshtein import distance as levenshtein
from lxml import etree
from tqdm.auto import tqdm


def pdf_to_txt(pdf_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    existing_txt_files = {f.stem for f in cache_dir.glob("*.txt")}

    num_previously_processed = 0

    for pdf_file in pdf_files:
        txt_file = cache_dir / f"{pdf_file.stem}.txt"
        if pdf_file.stem in existing_txt_files:
            num_previously_processed += 1
            continue
        try:
            text = ""
            with pymupdf.open(pdf_file) as doc:
                for page in doc:
                    text += page.get_text()
            txt_file.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
            traceback.print_exc()

    print(f"# of previously processed files: {num_previously_processed}")
    print(f"# of new files processed: {len(pdf_files) - num_previously_processed}")


def xml2text(path: pathlib.Path) -> str:
    root = etree.parse(str(path)).getroot()
    txt = " ".join(root.itertext())
    return txt


def clean_text(text: str) -> str:
    """
    Clean text by removing invisible Unicode characters that cause PDF extraction issues.
    """

    invisible_chars = [
        "\u00ad",  # Soft hyphen
        "\u200b",  # Zero width space
        "\u200c",  # Zero width non-joiner
        "\u200d",  # Zero width joiner
        "\u2060",  # Word joiner
        "\ufeff",  # Zero width no-break space (BOM)
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    # Normalize Unicode to canonical form (handles ligatures, accents, etc.)
    text = unicodedata.normalize("NFKC", text)

    # ascii
    # text = re.sub(r"[^\x00-\x7F]+", "", text)

    # doi conversion for zenodo
    text = re.sub(r"https?://zenodo\.org/record/(\d+)", r" 10.5281/zenodo.\1 ", text)

    return text


def load_data(data_dir: Path, output_dir: Path):
    cache_dir = output_dir / "exp_tmp" / "txt"

    pdf_dir = data_dir / "PDF"
    xml_dir = data_dir / "XML"

    print(f"Cache directory ({cache_dir}) exists? -> {cache_dir.exists()}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # run pdf to text ---
    pdf_to_txt(pdf_dir, cache_dir)

    records = []
    txt_files = list(cache_dir.glob("*.txt"))

    for txt_file in txt_files:
        id_ = txt_file.stem

        with open(txt_file, "r") as f:
            text = f.read()
        text = clean_text(text)
        records.append({"article_id": id_, "pdf_text": text})

    df = pd.DataFrame(records)

    # run xml to text ---
    article2xml = dict()
    for xml_path in xml_dir.glob("*.xml"):
        try:
            article2xml[xml_path.stem] = xml2text(xml_path)
        except Exception as e:
            print(f"XML error {xml_path.stem}: {e}")

    df["xml_text"] = df["article_id"].map(article2xml)
    df["xml_text"] = df["xml_text"].fillna("NA")

    return df


def split_text_on_tokens(text, tokenizer, tokens_per_chunk, chunk_overlap):
    splits = []
    input_ids = tokenizer.encode(text)

    start_idx = 0
    cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break

        start_idx += tokens_per_chunk - chunk_overlap
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))

        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


def _process_single_row(row_data, tokenizer, tokens_per_chunk, chunk_overlap):
    """Process a single row to generate chunks for both PDF and XML text."""
    aid, pdf_text, xml_text = row_data
    chunk_data = []

    # Process PDF text
    try:
        pdf_chunks = split_text_on_tokens(pdf_text, tokenizer, tokens_per_chunk, chunk_overlap)
        for idx, chunk in enumerate(pdf_chunks):
            chunk_data.append({"article_id": aid, "chunk": chunk, "source": "pdf", "chunk_idx": idx})
    except Exception as e:
        print(f"Error processing PDF for {aid}: {e}")

    # Process XML text
    if xml_text != "NA" and xml_text.strip():
        try:
            xml_chunks = split_text_on_tokens(xml_text, tokenizer, tokens_per_chunk, chunk_overlap)
            for idx, chunk in enumerate(xml_chunks):
                chunk_data.append({"article_id": aid, "chunk": chunk, "source": "xml", "chunk_idx": idx})
        except Exception as e:
            print(f"Error processing XML for {aid}: {e}")

    return chunk_data


def create_chunks(input_df, tokenizer, tokens_per_chunk=2048, chunk_overlap=256, n_jobs=8):
    row_data = [(row["article_id"], row["pdf_text"], row["xml_text"]) for _, row in input_df.iterrows()]

    # Process rows in parallel
    print(f"Processing {len(row_data)} articles using {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_process_single_row)(data, tokenizer, tokens_per_chunk, chunk_overlap) for data in tqdm(row_data, desc="Preparing parallel jobs"))

    chunk_data = []
    for result in results:
        chunk_data.extend(result)

    return pd.DataFrame(chunk_data)


def get_doi_prefix(x):
    if "https://doi.org/" not in x:
        return x
    return x.split("https://doi.org/")[-1].split("/")[0]


def enforce_doi_prefix_based_filter(doi):
    doi_prefix = get_doi_prefix(doi)
    if doi_prefix == "10.5256":  # f1000research
        if not re.search(r"\.d\d+$", doi):  # should end with .d<number>
            return False
    return True


def get_doi_hits(text, whitelist=None, **kwargs):
    DOI_LINK = "https://doi.org/"

    matches = re.findall(r'10\s*\.\s*\d{4,9}\s*/\s*[^\s()"<>&,#]+', text)

    if not matches:
        return []

    processed_matches = []
    for match in matches:
        dataset_id = re.sub(r"\s+", "", match).lower()
        dataset_id = re.sub(r"[^A-Za-z0-9]+$", "", dataset_id)
        processed_matches.append({"dataset_id": DOI_LINK + dataset_id, "match": match})

    grouped = defaultdict(list)

    for item in processed_matches:
        key = item["dataset_id"]
        grouped[key].append(item["match"])

    result = []

    if whitelist is not None:
        whitelist = set(whitelist)

    for dataset_id, match_list in grouped.items():
        dataset_prefix = get_doi_prefix(dataset_id)

        if whitelist is not None:
            if dataset_prefix not in whitelist:
                continue

        if not enforce_doi_prefix_based_filter(dataset_id):
            continue

        result.append({"dataset_id": dataset_id, "match_list": match_list, "pattern": "doi"})

    return result


def get_accession_hits(text, default_wsize=2000):
    """
    Extracts accession numbers from text using context-aware filtering rules.
    """

    rules = [
        # alphafold
        {
            "db": "alphafold",
            "pattern": r"\bAF-[OPQ][0-9][A-Z0-9]{3}[0-9]+-F[0-9]\b",
            "context": r"(?i)(alphafold|alphafold database|alphafold db|structures|predicted structure|predicted protein structure|protein|identifier|accession)",
            "wsize": default_wsize,
        },
        {
            "db": "alphafold",
            "pattern": r"\bAF-[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9])+-F[0-9]\b",
            "context": r"(?i)(alphafold|alphafold database|alphafold db|structures|predicted structure|predicted protein structure|protein|identifier|accession)",
            "wsize": default_wsize,
        },
        # arrayexpress
        {
            "db": "ArrayExpress",
            "pattern": r"\bE-[A-Z]{4}-\d+",
            "context": r"(?i)(arrayexpress|atlas|gxa|accession|experiment)",
            "wsize": default_wsize,
        },
        # bia (BioImage Archive)
        {
            "db": "bia",
            "pattern": r"\bS-BIAD\d+",
            "context": r"(?i)(bia|bioimage archive database|bioimage archive|database|identifier|accession)",
            "wsize": default_wsize,
        },
        # biomodels
        {
            "db": "biomodels",
            "pattern": r"\b(?:BIOMD|MODEL)\d{10}\b",
            "context": r"(?i)(biomodels|accession|model|identifier)",
            "wsize": default_wsize,
        },
        {
            "db": "biomodels",
            "pattern": r"\bBMID\d{12}\b",
            "context": r"(?i)(biomodels|accession|model|identifier)",
            "wsize": default_wsize,
        },
        # bioproject
        {
            "db": "BioProject",  # TP: 26 | FP: 22
            "pattern": r"\bPRJ[DEN][A-Z]\d+\b",
            "context": r"(?i)(bioproject|accession|archive)",
            "wsize": default_wsize,
        },
        # biosample TP: 41 | FP: 36
        {
            "db": "BioSample",
            "pattern": r"\bSAM[NED][A-Z]?\d+",
            "context": r"(?i)(biosample|accession|model)",
            "wsize": default_wsize,
        },
        # biostudies
        {
            "db": "biostudies",
            "pattern": r"\bS-[A-Z]{4}[A-Z0-9-]+\b",
            "context": None,
            "wsize": 0,
        },
        # cath
        {
            "db": "CATH",
            "pattern": r"\b[1-6]\.(?:[1-9]\d)\.(?:\d{1,4})\.(?:\d{1,4})\b",
            "context": r"(?i)(cath|cath-Gene3D|cath Gene3D|c.a.t.h|domain|families|cathnode|pdb|superfamily)",
            "wsize": default_wsize,
        },
        {
            "db": "CATH",
            "pattern": r"\b[0-9][a-zA-Z0-9]{3}[A-Z]\d{2}\b",
            "context": r"(?i)(cath|cath-Gene3D|cath Gene3D|c.a.t.h|domain|families|cathnode|pdb|superfamily)",
            "wsize": 100,
        },
        # cellosaurus
        {
            "db": "cellosaurus",
            "pattern": r"\bCVCL[_:\-][A-Z0-9]{4}\b",
            "context": r"(?i)(cells|cellosaurus|cellosaurus database|Cell lines|Cell Bank|cell lines|cell bank|accession number|RRID:)",
            "wsize": default_wsize,
        },
        # chembl
        {
            "db": "chembl",
            "pattern": r"\bCHEMBL\d+",
            "context": r"(?i)(chembl|compound)",
            "wsize": default_wsize,
        },
        # chebi
        {
            "db": "chebi",
            "pattern": r"\bCHEBI:\d+",
            "context": r"(?i)(chebi|compound)",
            "wsize": default_wsize,
        },
        # complexportal
        {
            "db": "complexportal",
            "pattern": r"\bCP\d{6}",
            "context": None,
            "wsize": 0,
        },
        {
            "db": "complexportal",
            "pattern": r"\bCPX-\d+",
            "context": r"(?i)(protein|complex)",
            "wsize": default_wsize,
        },
        # dbgap
        {
            "db": "dbgap",
            "pattern": r"\bphs\d{6}\b",
            "context": r"(?i)(database of genotypes and phenotypes|dbgap|accession|archives|studies|interaction)",
            "wsize": default_wsize,
        },
        # empiar
        {
            "db": "empiar",
            "pattern": r"\bEMPIAR-\d{5,}\b",
            "context": None,
            "wsize": 0,
        },
        # emdb
        {
            "db": "emdb",
            "pattern": r"\bEMD-\d{4,5}\b",
            "context": r"(?i)(emdb|accession|code)",
            "wsize": default_wsize,
        },
        # ega
        {
            "db": "ega",
            "pattern": r"\bEGA[SDAC]\d{11}\b",
            "context": r"(?i)(ega|accession|archive|studies|study|dataset|datasets|data set|data sets|validation sets|validation set|set|sets|data|dac|European Genome-phenome Archive|European Genome phenome Archive)",
            "wsize": default_wsize,
        },
        # ensembl
        {
            "db": "ensembl",
            "pattern": r"\b[eE][nN][sS][a-zA-Z]*[gG]\d{11,}\b",
            "context": r"(?i)(ensembl|accession|transcript|sequence)",
            "wsize": default_wsize,
        },
        {
            "db": "ensembl",
            "pattern": r"\b[eE][nN][sS][a-zA-Z]*[ptPT]\d{11,}\b",
            "context": r"(?i)(ensembl|accession|transcript|sequence)",
            "wsize": default_wsize,
        },
        # gen
        {
            "db": "gen",
            "pattern": r"\b[EDS]RP\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|study|studies)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[EDS]RX\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|experiment|experiments)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[EDS]RA\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|submission|submissions)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[EDS]RR\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|run|runs)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[EDS]RZ\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|analysis|analyses)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\bERS\d{5,}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|sample|samples)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[A-RT-Z][A-Z]{3}S?\d{8,9}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|protein coding|protein|sequence|sequences)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\b[A-Z]{3}\d{5}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|protein coding|protein|sequence|sequences)",
            "wsize": default_wsize,
        },
        {
            "db": "gen",
            "pattern": r"\bTI\d+\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|trace|traces)",
            "wsize": default_wsize,
        },
        {
            "db": "ena",
            "pattern": r"\b[A-Z]{2}\d{6}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|asssembled|annotated|sequence|sequences)",
            "wsize": 500,
        },
        {
            "db": "ena",
            "pattern": r"\b[A-Z]\d{5}\b",
            "context": r"(?i)(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|asssembled|annotated|sequence|sequences)",
            "wsize": 500,
        },
        # geo
        {
            "db": "geo",
            "pattern": r"\bG(?:PL|SM|SE|DS)\d{2,}\b",
            "context": r"(?i)(gene expression omnibus|genome|geo|accession|functional genomics|data repository|data submissions)",
            "wsize": default_wsize,
        },
        # gisaid
        {
            "db": "gisaid",
            "pattern": r"\bEPI_ISL_\d{6,}",
            "context": r"(?i)(gisaid|global initiative on sharing all influenza data|virus|viruses|strain|strains|sequence|sequences|flu|epiflu|identifier|database|accession)",
            "wsize": default_wsize,
        },
        {
            "db": "gisaid",
            "pattern": r"\bEPI\d{6,}",
            "context": r"(?i)(gisaid|global initiative on sharing all influenza data|segment|segments|identifier|flu|epi|epiflu|database|sequence|sequences|isolate|isolates|accession)",
            "wsize": default_wsize,
        },
        {
            "db": "gisaid",
            "pattern": r"\bEPI\d{6}-\d+",
            "context": r"(?i)(gisaid|global initiative on sharing all influenza data|segment|segments|identifier|flu|epi|epiflu|database|sequence|sequences|isolate|isolates|accession)",
            "wsize": default_wsize,
        },
        # hpa
        {
            "db": "hpa",
            "pattern": r"\b(?i:HPA)\d{6}",
            "context": None,
            "wsize": 0,
        },
        {
            "db": "hpa",
            "pattern": r"\b(?i:CAB)\d{6}",
            "context": None,
            "wsize": 0,
        },
        # intact
        {
            "db": "intact",
            "pattern": r"\bEBI-\d+",
            "context": r"(?i)(intact|IntAct|inTact|Intact|interaction|interactions|protein)",
            "wsize": default_wsize,
        },
        # interpro
        {
            "db": "interpro",
            "pattern": r"\bIPR\d{6}",
            "context": r"(?i)(interpro|domain|family|motif|accession)",
            "wsize": default_wsize,
        },
        # metabolights
        {
            "db": "metabolights",
            "pattern": r"\bMTBLS\d+",
            "context": r"(?i)(metabolights|accession|repository)",
            "wsize": default_wsize,
        },
        # metagenomics
        {
            "db": "metagenomics",
            "pattern": r"\bSRS\d{6}\b",
            "context": r"(?i)(samples|ebi metagenomics|metagenomics|database)",
            "wsize": default_wsize,
        },
        # orphadata
        {
            "db": "orphadata",
            "pattern": r"\bORPHA:\d+",
            "context": r"(?i)(database|rare disease|disease|nomenclature|data|syndrome|id|number|name|orphanet|orphadata|orpha)",
            "wsize": default_wsize,
        },
        {
            "db": "orphadata",
            "pattern": r"\bORPHA \d+",
            "context": r"(?i)(database|rare disease|disease|data|nomenclature|syndrome|id|number|name|orphanet|orphadata|orpha)",
            "wsize": default_wsize,
        },
        # pfam
        {
            "db": "pfam",
            "pattern": r"\bPF(AM)?\d{5}\b",
            "context": r"(?i)(pfam|domain|family|accession|motif)",
            "wsize": default_wsize,
        },
        # pxd
        {
            "db": "pxd",
            "pattern": r"\b(R)?PXD\d{6}\b",
            "context": r"(?i)(pxd|proteomexchange|pride|dataset|accession|repository)",
            "wsize": default_wsize,
        },
        # uniprot
        {
            "db": "UniProt",
            "pattern": r"\b(([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})|([OPQ][0-9][A-Z0-9]{3}[0-9]))(\.\d+)?\b",
            "context": r"(?i)(swiss-prot|sprot|uniprot|swiss prot|accession(s)?|Locus|GenBank|genome|sequence(s)?|protein|trembl|uniparc|uniprotkb|Acc.No|Acc. No)",
            "wsize": default_wsize,
        },
        # reactome
        {
            "db": "reactome",
            "pattern": r"\bR-HSA-\d+",
            "context": r"(?i)(biological|regulatory|pathway|pathways|database)",
            "wsize": default_wsize,
        },
        # refseq
        {
            "db": "refseq",
            "pattern": r"\b(((WP|AC|AP|NC|NG|NM|NP|NR|NT|NW|XM|XP|XR|YP|ZP)_\d+)|(NZ_[A-Z]{2,4}\d+))(\.\d+)?\b",
            "context": r"(?i)(refseq|genbank|accession|sequence|acc)",
            "wsize": 50,
        },
        # rfam
        {
            "db": "rfam",
            "pattern": r"\bRF\d{5}\b",
            "context": None,
            "wsize": 0,
        },
        # refsnp
        {
            "db": "refsnp",
            "pattern": r"\b[rs]s\d{1,9}\b",
            "context": r"(?i)(allele|model|multivariate|polymorphism|locus|loci|haplotype|genotype|variant|chromosome|SNPs|snp|snp(s)*)",
            "wsize": 50,
        },
        # rnacentral
        {
            "db": "rnacentral",
            "pattern": r"\bURS[0-9A-Z]+_\d+\b",
            "context": None,
            "wsize": 0,
        },
        # treefam
        {
            "db": "treefam",
            "pattern": r"\bTF\d{6}\b",
            "context": r"(?i)(treefam|tree|family|accession|dendrogram)",
            "wsize": default_wsize,
        },
        # uniparc
        {
            "db": "uniparc",
            "pattern": r"\bUPI[A-F0-9]{10}\b",
            "context": r"(?i)(uniprot|accession(s)?|Locus|sequence(s)?|protein|uniparc|Acc.No|Acc. No)",
            "wsize": default_wsize,
        },
        # igsr (1000genomes)
        {
            "db": "igsr",
            "pattern": r"\bHG0[0-4]\d{3}\b",
            "context": r"(\bcell\b|sample|iPSC|iPSCs|iPS|fibroblast|fibroblasts|QTL|eQTL|pluripotent|induced|\bdonor\b|\bstem\b|EBiSC|1000 Genomes|Coriell|\bLCL\b|lymphoblastoid)",
            "wsize": default_wsize,
        },
        {
            "db": "igsr",
            "pattern": r"\b(NA|GM)[0-2]\d{4}\b",
            "context": r"(\bcell\b|sample|iPSC|iPSCs|iPS|fibroblast|fibroblasts|QTL|eQTL|pluripotent|induced|\bdonor\b|\bstem\b|EBiSC|1000 Genomes|Coriell|\bLCL\b|lymphoblastoid)",
            "wsize": default_wsize,
        },
    ]

    all_matches = []
    match2pattern = {}

    for rule in rules:
        rule_type = rule["db"]
        accession_pattern = re.compile(rule["pattern"])
        context_pattern = rule["context"]
        window_size = rule["wsize"]

        for match in accession_pattern.finditer(text):
            accession_id = match.group(0)

            if not context_pattern:
                all_matches.append(accession_id)
                match2pattern[accession_id] = rule_type
                continue

            # # additional check for GENBANK ---
            if rule_type == "ena":
                win_doi = 4000
                start_doi = max(0, match.start() - win_doi)
                end_doi = min(len(text), match.end() + win_doi)
                context_window_doi = text[start_doi:end_doi]
                if re.search(r'10\s*\.\s*\d{4,9}\s*/\s*[^\s()"<>&,#]+', context_window_doi):
                    continue  # if doi in vicinity, skip

            # Otherwise, validate the match using the context window.
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            context_window = text[start:end]

            if re.search(context_pattern, context_window):
                all_matches.append(accession_id)
                match2pattern[accession_id] = rule_type

    if not all_matches:
        return []

    # The rest of the function remains the same for grouping and formatting results.
    grouped = defaultdict(list)
    group2pattern = defaultdict(list)
    for match in all_matches:
        dataset_id = re.sub(r"\s+", "", match)
        dataset_id = re.sub(r"[^A-Za-z0-9._-]+$", "", dataset_id)
        grouped[dataset_id].append(match)
        group2pattern[dataset_id].append(match2pattern[match])

    results = []
    for dataset_id, match_list in grouped.items():
        pattern_list = group2pattern[dataset_id]
        most_frequent_pattern = Counter(pattern_list).most_common(1)[0][0]
        results.append({"dataset_id": dataset_id, "match_list": match_list, "pattern": most_frequent_pattern})

    return results


def check_for_hit(text):
    doi_hits = get_doi_hits(text)
    acc_hits = get_accession_hits(text)
    return len(doi_hits + acc_hits) > 0


def num_doi_hits(text, whitelist=None, **kwargs):
    doi_hits = get_doi_hits(text, whitelist=whitelist)
    n = 0
    for hit in doi_hits:
        n += len(hit["match_list"])
    return n


def get_all_hits(text, whitelist=None, **kwargs):
    doi_hits = get_doi_hits(text, whitelist=whitelist)
    acc_hits = get_accession_hits(text)
    ids = [v["dataset_id"] for v in doi_hits] + [v["dataset_id"] for v in acc_hits]
    return ids


def run_regex(text, whitelist=None, **kwargs):
    doi_hits = get_doi_hits(text, whitelist=whitelist)
    acc_hits = get_accession_hits(text)

    ids = [v["dataset_id"] for v in doi_hits] + [v["dataset_id"] for v in acc_hits]
    patterns = [v["pattern"] for v in doi_hits] + [v["pattern"] for v in acc_hits]
    return {"dataset_ids": ids, "patterns": patterns}


DOI_LINK = "https://doi.org/"


def is_doi_link(name):
    return name.startswith(DOI_LINK)


def gt_dataset_id_normalization(name: str):
    if is_doi_link(name):
        return name.split(DOI_LINK)[-1].lower()
    return name.lower()


def check_presence(text, dataset_id):
    text = re.sub(r"\s+", "", text).lower()
    search_id = gt_dataset_id_normalization(dataset_id)
    return search_id in text


def _find_dataset_positions(text, dataset_id):
    positions = []
    search_id = gt_dataset_id_normalization(dataset_id)

    pattern_parts = []
    for char in search_id:
        if char in "./:":
            pattern_parts.append(r"\s*" + re.escape(char) + r"\s*")
        else:
            pattern_parts.append(re.escape(char))

    spaced_pattern = r"\s*".join(pattern_parts)

    for match in re.finditer(spaced_pattern, text, re.IGNORECASE):
        positions.append((match.start(), match.end()))

    positions = list(set(positions))
    positions.sort(key=lambda x: x[0])
    return positions


def get_window(text, dataset_id, tokenizer, max_tokens=2048, window_size_left=1536, window_size_right=512):
    positions = _find_dataset_positions(text, dataset_id)

    if not positions:
        return None

    positions.sort(key=lambda x: x[0])

    selected_sections = []

    for s, e in positions:
        start = max(0, s - window_size_left)
        end = min(e + window_size_right, len(text))
        selected_sections.append((start, end))

    merged_sections = []
    for start, end in selected_sections:
        if merged_sections and start <= merged_sections[-1][1]:
            merged_sections[-1] = (merged_sections[-1][0], max(merged_sections[-1][1], end))
        else:
            merged_sections.append((start, end))

    windows = []
    for start, end in merged_sections:
        windows.append(text[start:end])

    highlight = "\n\n".join([f"[Info {idx + 1}] {w} ..." for idx, w in enumerate(windows)])

    # truncate to max_tokens
    input_ids = tokenizer.encode(highlight)
    input_ids = input_ids[:max_tokens]
    highlight = tokenizer.decode(input_ids)

    return highlight


def get_prefix(x):
    return x.split("https://doi.org/")[-1].split("/")[0]


def add_candidates_from_online_index(input_df, index_df, candidate_df, pmc_acc_repos):
    blacklist_prefix = set(["10.6084", "10.5517", "10.17182"])

    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))
    dataset2repo_online = dict(zip(index_df["dataset_id"], index_df["repository"]))

    comp_articles = set(input_df["article_id"].values.tolist())
    new_candidates = []

    for aid in comp_articles:
        online_pool = article2dataset_online.get(aid, [])
        if len(online_pool) == 0:
            continue

        has_doi = any(c.startswith("https://doi.org") for c in online_pool)
        row = input_df[input_df["article_id"] == aid].iloc[0]

        for c in online_pool:
            if c.startswith("https://doi.org"):  # DOI
                # TODO: ignore figshare
                prefix = get_prefix(c)
                if prefix in blacklist_prefix:
                    continue

                if check_presence(row.pdf_text, c):
                    new_candidates.append({"article_id": aid, "dataset_id": c, "text": row.pdf_text, "source": "pdf", "pattern": "online"})

                if check_presence(row.xml_text, c):
                    new_candidates.append({"article_id": aid, "dataset_id": c, "text": row.xml_text, "source": "xml", "pattern": "online"})

            else:  # accession --
                if has_doi:
                    continue

                repo = dataset2repo_online.get(c, "NA")
                if repo in pmc_acc_repos:
                    if check_presence(row.pdf_text, c):
                        new_candidates.append({"article_id": aid, "dataset_id": c, "text": row.pdf_text, "source": "pdf", "pattern": "online"})
                    if check_presence(row.xml_text, c):
                        new_candidates.append({"article_id": aid, "dataset_id": c, "text": row.xml_text, "source": "xml", "pattern": "online"})

    new_candidate_df = pd.DataFrame(new_candidates)
    candidate_df = pd.concat([candidate_df, new_candidate_df]).drop_duplicates(subset=["article_id", "dataset_id"], keep="first").reset_index(drop=True)

    return candidate_df


def validate_online(dataset_id, pattern, online_pool, validate_acc=False):
    if len(online_pool) == 0:  # missing article in online corpus, keep
        return True

    if dataset_id.startswith("https://doi.org"):  # check for doi
        return dataset_id in online_pool

    if not validate_acc:
        return True

    # accession validation ---
    check_patterns = ["UniProt", "ArrayExpress", "BioProject", "CATH", "BioSample", "uniparc", "igsr", "alphafold", "ena"]
    if pattern in check_patterns:
        return dataset_id in online_pool

    return True


def filter_candidates_using_online_index(candidate_df, index_df, validate_acc=False):
    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))

    candidate_df["valid"] = candidate_df.apply(lambda row: validate_online(row["dataset_id"], row["pattern"], article2dataset_online.get(row["article_id"], []), validate_acc), axis=1)

    candidate_df = candidate_df[candidate_df["valid"]].reset_index(drop=True)
    candidate_df = candidate_df.drop(columns=["valid"])
    return candidate_df


def add_near_doi_misses_from_online_index(candidate_df, index_df, dist_th=2):
    blacklist_prefix = set(["10.6084", "10.5517", "10.17182"])
    candidate_cols = list(candidate_df.columns)

    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))

    cdf = candidate_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_candidate = dict(zip(cdf["article_id"], cdf["dataset_id"]))

    comp_articles = set(candidate_df["article_id"].values.tolist())
    new_predictions = []

    for aid in comp_articles:
        online_pool = article2dataset_online.get(aid, [])
        if len(online_pool) == 0:
            continue

        has_doi = any(c.startswith("https://doi.org") for c in online_pool)
        if not has_doi:
            continue

        candidate_pool = list(article2dataset_candidate.get(aid, []))
        if len(candidate_pool) == 0:
            continue

        for c in online_pool:
            if c.startswith("https://doi.org"):  # DOI
                prefix = get_prefix(c)
                if prefix in blacklist_prefix:
                    continue

                if c in candidate_pool:
                    continue

                # find minimum lavenstein distance between c and candidate_pool
                min_dist = min(levenshtein(c, cp) for cp in candidate_pool)
                closest_match = min(candidate_pool, key=lambda x: levenshtein(c, x))
                matched_row = candidate_df[(candidate_df["article_id"] == aid) & (candidate_df["dataset_id"] == closest_match)]

                # print(f"{c} no match found in {aid}, closest match: {closest_match} with distance {min_dist}")

                if min_dist <= dist_th:
                    new_example = {"article_id": aid, "dataset_id": c}
                    for col in candidate_cols:
                        if col not in new_example:
                            new_example[col] = matched_row[col].values[0]
                    new_predictions.append(new_example)

    new_predictions_df = pd.DataFrame(new_predictions)
    candidate_df = pd.concat([candidate_df, new_predictions_df]).drop_duplicates(subset=["article_id", "dataset_id"], keep="first").reset_index(drop=True)

    return candidate_df


def add_unseen_zenodo_hits(candidate_df, index_df):
    candidate_cols = list(candidate_df.columns)

    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))

    cdf = candidate_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_candidate = dict(zip(cdf["article_id"], cdf["dataset_id"]))

    comp_articles = set(candidate_df["article_id"].values.tolist())
    new_predictions = []

    for aid in comp_articles:
        online_pool = article2dataset_online.get(aid, [])
        if len(online_pool) == 0:
            continue

        has_zenodo_doi = any(c.startswith("https://doi.org/10.5281/zenodo.") for c in online_pool)
        if not has_zenodo_doi:
            continue

        candidate_pool = list(article2dataset_candidate.get(aid, []))
        if len(candidate_pool) == 0:
            continue

        for c in candidate_pool:
            if c.startswith("https://doi.org/10.5281/zenodo."):  # zenodo-DOI
                try:
                    zenodo_id = int(c.split("https://doi.org/10.5281/zenodo.")[-1])
                    prev_zenodo_id = zenodo_id - 1
                    prev_zenodo_doi = f"https://doi.org/10.5281/zenodo.{prev_zenodo_id}"

                    if prev_zenodo_doi in online_pool:
                        matched_row = candidate_df[(candidate_df["article_id"] == aid) & (candidate_df["dataset_id"] == c)]
                        new_example = {"article_id": aid, "dataset_id": prev_zenodo_doi}
                        for col in candidate_cols:
                            if col not in new_example:
                                new_example[col] = matched_row[col].values[0]
                        new_predictions.append(new_example)

                    # next zenodo-DOI
                    next_zenodo_id = zenodo_id + 1
                    next_zenodo_doi = f"https://doi.org/10.5281/zenodo.{next_zenodo_id}"
                    if next_zenodo_doi in online_pool:
                        matched_row = candidate_df[(candidate_df["article_id"] == aid) & (candidate_df["dataset_id"] == c)]
                        new_example = {"article_id": aid, "dataset_id": next_zenodo_doi}
                        for col in candidate_cols:
                            if col not in new_example:
                                new_example[col] = matched_row[col].values[0]
                        new_predictions.append(new_example)

                except Exception as e:
                    print(f"Error processing {c}: {e}")
                    continue

    new_predictions_df = pd.DataFrame(new_predictions)
    candidate_df = pd.concat([candidate_df, new_predictions_df]).drop_duplicates(subset=["article_id", "dataset_id"], keep="first").reset_index(drop=True)

    return candidate_df


def add_unseen_dryad_hits(candidate_df, index_df):
    candidate_cols = list(candidate_df.columns)

    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))

    cdf = candidate_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_candidate = dict(zip(cdf["article_id"], cdf["dataset_id"]))

    comp_articles = set(candidate_df["article_id"].values.tolist())
    new_predictions = []

    for aid in comp_articles:
        online_pool = article2dataset_online.get(aid, [])
        if len(online_pool) == 0:
            continue

        has_dryad_doi = any(c.startswith("https://doi.org/10.5061/dryad") for c in online_pool)
        if not has_dryad_doi:
            continue

        candidate_pool = list(article2dataset_candidate.get(aid, []))
        if len(candidate_pool) == 0:
            continue

        for c in candidate_pool:
            if c.startswith("https://doi.org/10.5061/dryad"):  # dryad-DOI
                try:
                    # online pool may contain f"{c}.1", f"{c}.2", etc.
                    pattern = c + r"\.\d+$"
                    # check for hits in the online pool
                    hits = [h for h in online_pool if re.match(pattern, h)]
                    if len(hits) == 0:
                        continue
                    for h in hits:
                        matched_row = candidate_df[(candidate_df["article_id"] == aid) & (candidate_df["dataset_id"] == c)]
                        new_example = {"article_id": aid, "dataset_id": h}
                        for col in candidate_cols:
                            if col not in new_example:
                                new_example[col] = matched_row[col].values[0]
                        new_predictions.append(new_example)

                except Exception as e:
                    print(f"Error processing {c}: {e}")
                    continue

    new_predictions_df = pd.DataFrame(new_predictions)
    candidate_df = pd.concat([candidate_df, new_predictions_df]).drop_duplicates(subset=["article_id", "dataset_id"], keep="first").reset_index(drop=True)

    return candidate_df


def get_doi_mapping(index_df, input_df):
    gdf = index_df.groupby("article_id")["dataset_id"].agg(set).reset_index()
    article2dataset_online = dict(zip(gdf["article_id"], gdf["dataset_id"]))

    article_to_row = {row["article_id"]: row for _, row in input_df.iterrows()}

    article2doi_online = {}
    for article_id, dataset_ids in article2dataset_online.items():
        if article_id not in article_to_row:
            continue

        doi_candidates = [y for y in dataset_ids if y.startswith("https://doi.org")]

        row = article_to_row[article_id]
        filtered_dois = []

        for doi in doi_candidates:
            if check_presence(row["text"], doi):
                filtered_dois.append(doi)
            article2doi_online[article_id] = filtered_dois

    return article2doi_online


def split_text(text, char_per_chunk=1024, char_overlap=64):
    """Split text into overlapping chunks."""
    splits = []
    start_idx = 0
    cur_idx = min(start_idx + char_per_chunk, len(text))

    while start_idx < len(text):
        splits.append(text[start_idx:cur_idx])
        if cur_idx == len(text):
            break
        start_idx += char_per_chunk - char_overlap
        cur_idx = min(start_idx + char_per_chunk, len(text))

    return splits


def check_hit(text, keyword):
    text = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
    keyword = re.sub(r"[^a-zA-Z0-9]", "", keyword).lower()
    return keyword in text


def get_token_count(text, tokenizer):
    """Count the number of tokens in text using the given tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def score_context(context):
    top_tier = ["data availability", "data archiving", "data access", "data deposition", "data accessibility", "database description", "data deposition", "data resources", "data availability statement", "data archiving statement", "supporting information"]
    mid_tier = [
        "data collection",
        "data available",
        "publicly available at",
        "supplemental material",
        "mendeley data",
        "data repository",
        "retrieved from",
        "downloaded from",
        "data are available",
        "are available from",
        "deposited data",
        "deposited at",
        "available at",
        "data sharing",
    ]
    low_tier = ["accession number", "archived in", "uploaded to", "raw data", "acknowledgement", "freely available", "submitted to", "deposited in", "open data"]

    scoring_schema = {}
    for k in low_tier:
        scoring_schema[k] = 1.0
    for k in mid_tier:
        scoring_schema[k] = 3.0
    for k in top_tier:
        scoring_schema[k] = 5.0

    score = 0.0
    for k, v in scoring_schema.items():
        if check_hit(context.lower(), k.lower()):
            score += v

    return score


def find_and_decorate(text, dataset_id):
    positions = _find_dataset_positions(text, dataset_id)

    if not positions:
        return text

    positions.sort(key=lambda x: x[0])

    result = text
    offset = 0

    for start, end in positions:
        adj_start = start + offset
        adj_end = end + offset

        matched_text = text[start:end]
        wrapped_text = f"<b><focus> {matched_text} </focus></b>"

        result = result[:adj_start] + wrapped_text + result[adj_end:]

        offset += len(wrapped_text) - (end - start)

    return result


def get_context(cfg, row, tokenizer):
    """
    Extract relevant context for a dataset mention from article text.
    Fixed version addressing logical issues.
    """

    dataset_id = row["dataset_id"]
    detected_dois = row.get("detected_dois", []) or []  # Handle missing/None detected_dois
    text = row["text"]

    chunks = split_text(text, char_per_chunk=cfg.context.char_per_chunk, char_overlap=cfg.context.char_overlap)
    n_chunks = len(chunks)

    scores = [score_context(c) for c in chunks]
    idx2score = {idx: s for idx, s in enumerate(scores)}

    # Get top scoring chunks
    top_k_idxs = np.argsort(scores)[-cfg.context.top_k :][::-1]
    top_chunks = []
    for idx in top_k_idxs:
        if scores[idx] >= cfg.context.relevance_th:
            top_chunks.append(chunks[idx])

    # Get chunks containing dataset_id mentions
    hit_chunks = []
    first_hit_check = False
    for idx, chunk in enumerate(chunks):
        if check_presence(chunk, dataset_id):
            if not first_hit_check:
                prev_chunk = chunks[max(0, idx - 1)]
                if prev_chunk not in hit_chunks:
                    hit_chunks.append(prev_chunk)
                if chunk not in hit_chunks:
                    hit_chunks.append(chunk)
                next_chunk = chunks[min(idx + 1, n_chunks - 1)]
                if next_chunk not in hit_chunks:
                    hit_chunks.append(next_chunk)
            else:
                first_hit_check = True
                if chunk not in hit_chunks:
                    hit_chunks.append(chunk)

    # Get chunks containing DOIs (only if detected_dois is not empty)
    doi_chunks = []
    if detected_dois:  # Validation added
        for idx, chunk in enumerate(chunks):
            if any([check_presence(chunk, d) for d in detected_dois]):
                doi_chunks.append((chunk, idx2score[idx]))

        doi_chunks = sorted(doi_chunks, key=lambda x: x[1], reverse=True)[: cfg.context.max_doi_chunk]
        doi_chunks = [c[0] for c in doi_chunks]

    # Collect all candidate chunks in priority order
    seen_chunks = set()
    selected_chunks = []
    num_tokens = 0

    for chunk in hit_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    if chunks and chunks[0] not in seen_chunks:
        first_chunk_tokens = get_token_count(chunks[0], tokenizer)
        if num_tokens + first_chunk_tokens <= cfg.context.token_budget:
            seen_chunks.add(chunks[0])
            selected_chunks.append(chunks[0])
            num_tokens += first_chunk_tokens

    for chunk in doi_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    for chunk in top_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    # Find positions of selected chunks in original text and merge overlapping spans
    spans = []
    for chunk in selected_chunks:
        pos = text.find(chunk)
        if pos != -1:
            spans.append((pos, pos + len(chunk)))

    spans.sort()
    merged_sections = []

    for start, end in spans:
        if merged_sections and start <= merged_sections[-1][1]:
            merged_sections[-1] = (merged_sections[-1][0], max(merged_sections[-1][1], end))
        else:
            merged_sections.append((start, end))

    windows = []
    for start, end in merged_sections:
        section = text[start:end].strip()
        if section:
            windows.append(section)

    context = "\n\n".join([f"[Section {idx + 1}]\n{w}" for idx, w in enumerate(windows)])
    context = find_and_decorate(context, dataset_id)

    return context


def get_context_v2(cfg, row, tokenizer):
    """
    Extract relevant context for a dataset mention from article text.
    Fixed version addressing logical issues.
    """

    dataset_id = row["dataset_id"]
    detected_dois = row.get("detected_dois", []) or []  # Handle missing/None detected_dois
    text = row["text"]

    chunks = split_text(text, char_per_chunk=cfg.context.char_per_chunk, char_overlap=cfg.context.char_overlap)
    n_chunks = len(chunks)

    scores = [score_context(c) for c in chunks]
    idx2score = {idx: s for idx, s in enumerate(scores)}

    # Get top scoring chunks
    top_k_idxs = np.argsort(scores)[-cfg.context.top_k :][::-1]
    top_chunks = []
    for idx in top_k_idxs:
        if scores[idx] >= cfg.context.relevance_th:
            top_chunks.append(chunks[idx])

    # Get chunks containing dataset_id mentions
    hit_chunks = []
    first_hit_check = False
    for idx, chunk in enumerate(chunks):
        if check_presence(chunk, dataset_id):
            if not first_hit_check:
                prev_chunk = chunks[max(0, idx - 1)]
                if prev_chunk not in hit_chunks:
                    hit_chunks.append(prev_chunk)
                if chunk not in hit_chunks:
                    hit_chunks.append(chunk)
                next_chunk = chunks[min(idx + 1, n_chunks - 1)]
                if next_chunk not in hit_chunks:
                    hit_chunks.append(next_chunk)
            else:
                first_hit_check = True
                if chunk not in hit_chunks:
                    hit_chunks.append(chunk)

    # Get chunks containing DOIs (only if detected_dois is not empty)
    doi_chunks = []
    if detected_dois:  # Validation added
        for idx, chunk in enumerate(chunks):
            if any([check_presence(chunk, d) for d in detected_dois]):
                doi_chunks.append((chunk, idx2score[idx]))

        doi_chunks = sorted(doi_chunks, key=lambda x: x[1], reverse=True)[: cfg.context.max_doi_chunk]
        doi_chunks = [c[0] for c in doi_chunks]

    # Collect all candidate chunks in priority order
    seen_chunks = set()
    selected_chunks = []
    num_tokens = 0

    if chunks and chunks[0] not in seen_chunks:
        first_chunk_tokens = get_token_count(chunks[0], tokenizer)
        if num_tokens + first_chunk_tokens <= cfg.context.token_budget:
            seen_chunks.add(chunks[0])
            selected_chunks.append(chunks[0])
            num_tokens += first_chunk_tokens

    for chunk in doi_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    for chunk in hit_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    for chunk in top_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    # Find positions of selected chunks in original text and merge overlapping spans
    spans = []
    for chunk in selected_chunks:
        pos = text.find(chunk)
        if pos != -1:
            spans.append((pos, pos + len(chunk)))

    spans.sort()
    merged_sections = []

    for start, end in spans:
        if merged_sections and start <= merged_sections[-1][1]:
            merged_sections[-1] = (merged_sections[-1][0], max(merged_sections[-1][1], end))
        else:
            merged_sections.append((start, end))

    windows = []
    for start, end in merged_sections:
        section = text[start:end].strip()
        if section:
            windows.append(section)

    context = "\n\n".join([f"[Section {idx + 1}]\n{w}" for idx, w in enumerate(windows)])
    context = find_and_decorate(context, dataset_id)

    return context


def detect_family(chunk, family_ids):
    for fid in family_ids:
        if check_presence(chunk, fid):
            return True
    return False


def get_context_many_accessions(cfg, row, tokenizer):
    dataset_id = row["dataset_id"]
    detected_dois = row.get("detected_dois", []) or []
    detected_accessions = row.get("detected_accession_ids", []) or []

    if len(detected_accessions) <= cfg.max_accession_ids_per_family:
        return get_context_v2(cfg, row, tokenizer)

    text = row["text"]

    chunks = split_text(text, char_per_chunk=cfg.context.char_per_chunk, char_overlap=cfg.context.char_overlap)
    n_chunks = len(chunks)

    scores = [score_context(c) for c in chunks]
    idx2score = {idx: s for idx, s in enumerate(scores)}

    # Get top scoring chunks
    top_k_idxs = np.argsort(scores)[-cfg.context.top_k :][::-1]
    top_chunks = []
    for idx in top_k_idxs:
        if scores[idx] >= cfg.context.relevance_th:
            top_chunks.append(chunks[idx])

    hit_chunks = []
    for idx, chunk in enumerate(chunks):
        if check_presence(chunk, dataset_id):
            start_idx = idx
            while start_idx > 0 and detect_family(chunks[start_idx - 1], detected_accessions):
                start_idx -= 1

            end_idx = idx
            while end_idx < n_chunks - 1 and detect_family(chunks[end_idx + 1], detected_accessions):
                end_idx += 1

            # start_idx = max(0, start_idx+1)
            # if start_idx < idx and chunks[start_idx] not in hit_chunks:
            #     hit_chunks.append(chunks[start_idx])

            if start_idx + 1 < idx and chunks[start_idx + 1] not in hit_chunks:
                hit_chunks.append(chunks[start_idx + 1])

            if chunk not in hit_chunks:
                hit_chunks.append(chunk)

            # if end_idx > idx and chunks[end_idx] not in hit_chunks:
            #     hit_chunks.append(chunks[end_idx])

            if end_idx - 1 > idx and chunks[end_idx - 1] not in hit_chunks:
                hit_chunks.append(chunks[end_idx - 1])

    doi_chunks = []
    if detected_dois:
        for idx, chunk in enumerate(chunks):
            if any([check_presence(chunk, d) for d in detected_dois]):
                doi_chunks.append((chunk, idx2score[idx]))

        doi_chunks = sorted(doi_chunks, key=lambda x: x[1], reverse=True)[: cfg.context.max_doi_chunk]
        doi_chunks = [c[0] for c in doi_chunks]

    # Collect all candidate chunks in priority order
    seen_chunks = set()
    selected_chunks = []
    num_tokens = 0

    # Add first chunk if within budget
    if chunks and chunks[0] not in seen_chunks:
        first_chunk_tokens = get_token_count(chunks[0], tokenizer)
        if num_tokens + first_chunk_tokens <= cfg.context.token_budget:
            seen_chunks.add(chunks[0])
            selected_chunks.append(chunks[0])
            num_tokens += first_chunk_tokens

    # Add DOI chunks
    for chunk in doi_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    # Add hit chunks
    for chunk in hit_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    # Add top chunks
    for chunk in top_chunks:
        if chunk not in seen_chunks:
            n_tok = get_token_count(chunk, tokenizer)
            if num_tokens + n_tok <= cfg.context.token_budget:
                seen_chunks.add(chunk)
                selected_chunks.append(chunk)
                num_tokens += n_tok

    # Find positions of selected chunks in original text and merge overlapping spans
    spans = []
    for chunk in selected_chunks:
        pos = text.find(chunk)
        if pos != -1:
            spans.append((pos, pos + len(chunk)))

    spans.sort()
    merged_sections = []

    for start, end in spans:
        if merged_sections and start <= merged_sections[-1][1]:
            merged_sections[-1] = (merged_sections[-1][0], max(merged_sections[-1][1], end))
        else:
            merged_sections.append((start, end))

    windows = []
    for start, end in merged_sections:
        section = text[start:end].strip()
        if section:
            windows.append(section)

    context = "\n\n".join([f"[Section {idx + 1}]\n{w}" for idx, w in enumerate(windows)])
    context = find_and_decorate(context, dataset_id)

    return context
