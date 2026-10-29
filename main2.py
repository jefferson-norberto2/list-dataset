"""Script de avaliação de similaridade (SSIM) entre imagens de referência e distorcidas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset.analyzer import evaluate_dataset_similarity

# Caminhos padrão do dataset CSIQ
DEFAULT_REF_DIR = Path("/home/jmn/Pictures/CSIQ/src_imgs")
DEFAULT_DIST_DIR = Path("/home/jmn/Pictures/CSIQ/dst_imgs")
DEFAULT_CSV_OUTPUT = Path("csiq_similarity_levels.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avalia similaridade SSIM entre imagens originais e distorcidas por nível.",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=DEFAULT_REF_DIR
        if DEFAULT_REF_DIR.is_dir()
        else Path("Datasets/src_imgs"),
        help="Diretório das imagens originais/referência.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR
        if DEFAULT_DIST_DIR.is_dir()
        else Path("Datasets/dst_imgs"),
        help="Diretório das imagens distorcidas (busca recursiva).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Caminho do arquivo CSV para exportar os resultados detalhados (padrão: csiq_similarity_levels.csv).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Não salvar arquivo CSV com os resultados individuais.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos paralelos (padrão: automático pelo nproc).",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"Diretório de referência: {args.ref_dir}")
    print(f"Diretório de distorções: {args.dist_dir}")

    # Executa a avaliação completa calculando SSIM, média, mediana e diferenças por nível
    report = evaluate_dataset_similarity(
        ref_dir=args.ref_dir,
        dist_dir=args.dist_dir,
        max_workers=args.workers,
        show_progress=True,
    )

    # Exibe o resumo formatado com média, mediana e conclusão descritiva
    print("\n" + report.format_summary())

    # Exporta para CSV
    if not args.no_csv and report.results:
        report.save_csv(args.csv)
        print(f"\nResultados detalhados salvos com sucesso em: {args.csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
