"""按匿名 StudyInstanceUID 下载检查子集（引用闭包自动补全）。

数据来源严格限定为“已完成任务的成功匿名化输出记录”（result.files 中
status != error 且记录了 output_name / 匿名 SOP UID 的条目），
**不重新解压或处理原始 ZIP**：

1. 以任务成功输出记录中匿名 StudyInstanceUID 对应的全部实例作为初选集合，
   并以同一份记录建立匿名 SOPInstanceUID -> 输出文件的引用索引
   （**不预先按磁盘存在性过滤**，避免丢片后初选悄悄少文件而产出残缺 ZIP）；
2. 递归读取集合内匿名 DICOM 任意序列层级的 ReferencedSOPInstanceUID，
   把同任务内被引用的实例（可能属于其他检查）补入集合，循环至闭包稳定；
3. 初选实例与每一个递归补入实例在读取前都核对物理文件是否仍在 output_dir；
   引用目标不在成功输出记录中、一个匿名 SOP UID 对应多个输出文件、
   或记录存在但物理文件已丢失时，一律拒绝输出（交由路由层返回 409
   与具体匿名 UID/原因），不产生残缺 ZIP。
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional

import pydicom
from pydicom.sequence import Sequence

from .anonymizer import FileReport, JobResult

REFERENCED_SOP_KEYWORD = "ReferencedSOPInstanceUID"

# 409 问题原因
REASON_NOT_IN_OUTPUT = "not_in_output"
REASON_AMBIGUOUS_OUTPUT = "ambiguous_output"
REASON_OUTPUT_MISSING = "output_missing"

# 同一匿名 SOP UID 命中多种问题时的原因优先级：物理文件缺失最严重，
# 一旦该 SOP 有任一记录文件丢失，歧义列表本身也无法完整核对
_REASON_PRIORITY = {
    REASON_OUTPUT_MISSING: 3,
    REASON_AMBIGUOUS_OUTPUT: 2,
    REASON_NOT_IN_OUTPUT: 1,
}

_PROBLEM_MESSAGES = {
    REASON_NOT_IN_OUTPUT: "引用目标不在本任务的成功输出中",
    REASON_AMBIGUOUS_OUTPUT: "一个匿名 SOPInstanceUID 对应多个输出文件，无法唯一补入",
    REASON_OUTPUT_MISSING: "任务记录有该成功输出，但 output_dir 中的物理文件已不存在",
}


class StudySubsetError(Exception):
    """子集解析失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        study_uid: Optional[str] = None,
        problems: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.detail: dict = {"code": code, "message": message}
        if study_uid is not None:
            self.detail["study_instance_uid"] = study_uid
        if problems:
            self.detail["problems"] = problems


@dataclass
class StudyClosure:
    initial_reports: list[FileReport]
    included_reports: list[FileReport]

    @property
    def initial_count(self) -> int:
        return len(self.initial_reports)

    @property
    def added_count(self) -> int:
        initial_names = {r.output_name for r in self.initial_reports}
        return sum(r.output_name not in initial_names for r in self.included_reports)

    @property
    def included_output_names(self) -> list[str]:
        return sorted(r.output_name for r in self.included_reports)


# --------------------------------------------------------------------------- #
# 索引与引用遍历
# --------------------------------------------------------------------------- #
def _output_path(result: JobResult, report: FileReport) -> str:
    return os.path.join(result.output_dir, report.output_name)


def _successful_outputs(result: JobResult) -> list[FileReport]:
    """任务的成功匿名化输出**记录**（error/无输出名/无 SOP UID 者剔除）。

    注意：不在这里做磁盘存在性检查——初选与引用索引必须完整反映任务的
    成功输出记录，物理文件缺失在闭包解析阶段统一核对并报 output_missing，
    否则丢片会被静默过滤，进而产出缺片 ZIP。
    """
    return [
        report
        for report in result.files
        if report.status != "error"
        and report.output_name
        and report.sop_instance_uid
    ]


def _build_sop_index(result: JobResult) -> dict[str, list[FileReport]]:
    """匿名 SOPInstanceUID -> 成功输出记录（通常 1 个；重复 SOP 时多个）。

    仅依据任务记录建索引，不预过滤磁盘文件：被引用目标的物理文件若已
    丢失，应报 output_missing 而非误判为 not_in_output。
    """
    index: dict[str, list[FileReport]] = {}
    for report in _successful_outputs(result):
        index.setdefault(report.sop_instance_uid, []).append(report)
    return index


def _split_ui_values(value: object):
    """展开 ReferencedSOPInstanceUID 的单值 / VM>1（反斜杠分隔）/ 多值形式。"""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split("\\")
    for part in parts:
        text = str(part).strip()
        if text:
            yield text


def iter_referenced_sop_uids(ds) -> list[str]:
    """收集匿名 DICOM 中任意序列层级的全部 ReferencedSOPInstanceUID。"""
    refs: list[str] = []
    stack = [ds]
    while stack:
        current = stack.pop()
        for tag_ in list(current.keys()):
            elem = current[tag_]
            if elem.VR == "SQ" and isinstance(elem.value, Sequence):
                stack.extend(elem.value)
                continue
            if (getattr(elem, "keyword", "") or "") == REFERENCED_SOP_KEYWORD:
                refs.extend(_split_ui_values(elem.value))
    return refs


# --------------------------------------------------------------------------- #
# 引用闭包
# --------------------------------------------------------------------------- #
def _record_problem(
    problems: dict[str, tuple[str, list[str]]],
    uid: str,
    reason: str,
    output_files: Optional[list[str]] = None,
) -> None:
    """登记某个匿名 SOP UID 的问题；同 UID 多问题按优先级保留最严重者。

    output_files 随原因累加去重（如同一 UID 多份记录文件先后发现丢失）。
    """
    files = sorted(output_files or [])
    existing = problems.get(uid)
    if existing is None or _REASON_PRIORITY[reason] > _REASON_PRIORITY[existing[0]]:
        problems[uid] = (reason, files)
    elif reason == existing[0]:
        merged = sorted(set(existing[1]) | set(files))
        problems[uid] = (reason, merged)


def resolve_study_closure(result: JobResult, study_uid: str) -> StudyClosure:
    """返回某匿名检查在本任务成功输出记录上的引用闭包；无法闭合时抛 StudySubsetError。"""
    outputs = _successful_outputs(result)
    index = _build_sop_index(result)

    initial = [
        report for report in outputs if report.study_instance_uid == study_uid
    ]
    if not initial:
        raise StudySubsetError(
            "STUDY_NOT_FOUND",
            404,
            "该匿名 StudyInstanceUID 在本任务的成功输出中不存在",
            study_uid=study_uid,
        )

    chosen: dict[str, FileReport] = {r.output_name: r for r in initial}
    queue: list[FileReport] = list(initial)
    queued = {r.output_name for r in initial}
    # uid -> (reason, output_files)；同一 UID 只报一次，按优先级保留最严重判定
    problems: dict[str, tuple[str, list[str]]] = {}

    while queue:
        report = queue.pop()
        path = _output_path(result, report)

        # 初选与递归补入的实例逐个核对物理文件：记录有、文件无即 output_missing，
        # 不读取、不扫描其引用（文件内容已不可信），但继续遍历以汇总全部问题
        if not os.path.isfile(path):
            _record_problem(
                problems,
                report.sop_instance_uid,
                REASON_OUTPUT_MISSING,
                [report.output_name],
            )
            continue

        try:
            ds = pydicom.dcmread(path)
        except Exception as exc:  # 自己写出的匿名文件不可读属于服务端异常
            raise StudySubsetError(
                "ANONYMIZED_OUTPUT_UNREADABLE",
                500,
                f"无法读取匿名输出文件 {report.output_name}：{exc}",
                study_uid=study_uid,
            ) from exc

        # 闭包成员自身的匿名 SOP UID 若对应多个输出文件，同样属于歧义：
        # 不挑选、不猜测，直接 409（全量下载行为不受影响）
        own_targets = index.get(report.sop_instance_uid, [])
        if len(own_targets) > 1:
            _record_problem(
                problems,
                report.sop_instance_uid,
                REASON_AMBIGUOUS_OUTPUT,
                [t.output_name for t in own_targets],
            )

        for ref_uid in iter_referenced_sop_uids(ds):
            targets = index.get(ref_uid, [])
            if not targets:
                _record_problem(problems, ref_uid, REASON_NOT_IN_OUTPUT)
                continue
            if len(targets) > 1:
                _record_problem(
                    problems,
                    ref_uid,
                    REASON_AMBIGUOUS_OUTPUT,
                    [t.output_name for t in targets],
                )
            # 即使存在歧义也继续向下扫描，尽量一次性暴露全部闭包问题；
            # 有 problems 时最终必定 409，chosen 仅用于遍历，不会被输出。
            # 引用目标若仅物理文件丢失（记录仍在索引中），在轮到它出队时
            # 统一核对为 output_missing，而不是被误判为 not_in_output。
            for target in targets:
                chosen.setdefault(target.output_name, target)
                if target.output_name not in queued:
                    queued.add(target.output_name)
                    queue.append(target)

    if problems:
        problem_items = [
            {
                "sop_instance_uid": uid,
                "reason": reason,
                "output_files": output_files,
                "message": _PROBLEM_MESSAGES[reason],
            }
            for uid, (reason, output_files) in sorted(problems.items())
        ]
        raise StudySubsetError(
            "REFERENCE_CLOSURE_INCOMPLETE",
            409,
            "引用闭包无法在本任务的成功输出内闭合，已拒绝输出子集 ZIP",
            study_uid=study_uid,
            problems=problem_items,
        )

    # 无 problems 时闭包内每个实例均已通过物理文件核对，ZIP 不会残缺
    return StudyClosure(
        initial_reports=initial,
        included_reports=[chosen[name] for name in sorted(chosen)],
    )


# --------------------------------------------------------------------------- #
# 扁平 ZIP（复用匿名输出文件，不再触碰原始 ZIP / 解压目录）
# --------------------------------------------------------------------------- #
def build_subset_zip(output_dir: str, output_names: list[str], dest_zip: str) -> str:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in output_names:  # 调用方已排序；arcname 扁平、即匿名文件名
            zf.write(os.path.join(output_dir, name), arcname=name)
    return dest_zip


def write_study_subset_zip(job, study_uid: str) -> tuple[str, int, int]:
    """构建子集 ZIP 到任务目录下的临时文件，返回 (路径, 初选数, 补入数)。"""
    result: JobResult = job.result
    closure = resolve_study_closure(result, study_uid)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"study-subset-{job.id[:8]}-",
        suffix=".zip",
        dir=os.path.dirname(result.output_dir),
    )
    os.close(fd)
    try:
        build_subset_zip(result.output_dir, closure.included_output_names, tmp_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path, closure.initial_count, closure.added_count
