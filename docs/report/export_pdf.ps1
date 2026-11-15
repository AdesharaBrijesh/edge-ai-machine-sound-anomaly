# Update the table of contents in the generated DOCX with Microsoft Word and export a PDF.
#   powershell -ExecutionPolicy Bypass -File docs/report/export_pdf.ps1
param(
  [string]$Docx = (Join-Path $PSScriptRoot 'Pump_Sound_Anomaly_Detection_Report.docx')
)
$ErrorActionPreference = 'Stop'
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, '.pdf')

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($Docx, $false, $false)
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $null = $doc.Fields.Update()
  $doc.Save()
  $doc.SaveAs([ref]$Pdf, [ref]17)   # 17 = wdFormatPDF
  $pages = $doc.ComputeStatistics(2) # 2 = wdStatisticPages
  $doc.Close([ref]0)
  Write-Output "updated $Docx"
  Write-Output "wrote $Pdf ($pages pages)"
}
finally {
  $word.Quit()
  [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
