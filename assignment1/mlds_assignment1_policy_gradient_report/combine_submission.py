"""Optionally join the existing shuttle report and policy-gradient report.
Requires pypdf. Resolve named links before merging to avoid section-name collisions.
"""
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
root=Path(__file__).resolve().parent
sources=[(root.parent/'mlds_assignment1_shuttle/SHUTTLE_REPORT.pdf','Shuttle dispatch'),
         (root/'POLICY_GRADIENT_REPORT.pdf','Policy gradient: CartPole and Pong')]
writer=PdfWriter()
for source,title in sources:
 reader=PdfReader(source)
 destinations=reader.named_destinations
 for page in reader.pages:
  for ref in page.get('/Annots',[]):
   annotation=ref.get_object()
   if '/A' in annotation:
    action=annotation['/A']
    if action.get('/S')=='/GoTo' and isinstance(action.get('/D'),str):
     destination=destinations.get(action['/D'])
     if destination is not None:action[NameObject('/D')]=destination.dest_array
   if isinstance(annotation.get('/Dest'),str):
    destination=destinations.get(annotation['/Dest'])
    if destination is not None:annotation[NameObject('/Dest')]=destination.dest_array
 writer.append(reader,outline_item=title)
writer.add_metadata({'/Title':'MLDS Assignment 1: Shuttle Dispatch, CartPole and Pong','/Author':'FailedMath3U'})
output=root/'ASSIGNMENT1_FULL_REPORT.pdf'
with output.open('wb') as f:writer.write(f)
print(f'Saved {output}: {len(writer.pages)} pages')
