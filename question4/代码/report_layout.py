"""Route generated reports to the organized document tree."""
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parent.parent


def document_path(name):
    if name == '问题4.md':
        return ROOT/'第四问解答.md'
    if name.startswith('问题4_') and name.endswith('.md'):
        return ROOT/'资料'/name
    raise ValueError(f'Unknown generated document: {name}')


def normalize_report(text):
    def link(match):
        value = match[2].strip('<>')
        if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', value) or value.startswith('#'):
            return match[0]
        path, sep, anchor = value.partition('#')
        target = Path(path)
        if not target.is_absolute():
            parts = target.parts
            if parts and parts[0] == '问题4结果':
                target = ROOT/'数据'/'实验结果'/Path(*parts[1:])
            elif len(parts) == 1 and path.endswith('.md'):
                target = document_path(path)
            elif len(parts) == 1 and path.endswith(('.py', '.txt', '.html')):
                target = ROOT/'代码'/path
            else:
                target = ROOT/target
        target = Path(os.path.normpath(target))
        suffix = '#'+anchor if sep else ''
        return f'{match[1]}({target}{suffix})'
    text = re.sub(r'(!?\[[^\]\n]*\])\(([^)\n]+)\)', link, text)
    text = re.sub(r'B题/solution/(?=[a-zA-Z0-9_]+\.(?:py|txt))',
                  'B题/solution/代码/', text)
    text = text.replace('在'+chr(96)+'B题/solution'+chr(96),
                        '在'+chr(96)+'B题/solution/代码'+chr(96))
    text = text.replace('（在本目录）', '（在代码目录）')
    if re.search(r'python(?:3)? ', text):
        lines = text.splitlines()
        lines[1:1] = ['', f'> 裸文件名运行命令均在[代码目录]({ROOT}/代码)执行。']
        text = '\n'.join(lines)+'\n'
    return text


def write_report(name, text):
    target = document_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalize_report(text), encoding='utf-8')
    return target
