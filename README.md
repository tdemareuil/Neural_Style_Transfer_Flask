# Neural_style_transfer_Flask

I built this simple app using `flask` to perform Neural Style Transfer (NST). You can download/clone the repository and run it locally.

In this repository you'll also find a notebook version of the model for easier testing.

The app and models were inspired by several blog posts ([1], [2], [3], [4]), which themselves were based on the original NST [research article](https://arxiv.org/abs/1508.06576) by _Gatys et al._ (2016).

---

Examples of results:

* Pointillist painting style (Paul Signac, [_La Baie (Saint-Tropez)_](https://www.christies.com/lotfinder/Lot/paul-signac-1863-1935-la-baie-saint-tropez-6202464-details.aspx), 1907):

<table>
<tr>
  <td align='center' colspan=2> <strong> Style
<tr>
  <td align='center' colspan=2> <img src="static/image/outputs/s5.jpg" width="350" title="Style"> 
<tr>
  <th>Input <th> Output
<tr>
  <td> <img src="static/image/outputs/IMG_20170618_005324 - copie.jpg" width="350" title="Input">
  <td> <img src="static/image/outputs/output.png" width="350" title="Output">

<!-- Other way to build a table (Github Flavored Markdown, less flexible):
Input | Style | Output
:---:|:---:|:---:
<img src="static/image/outputs/IMG_20170618_005324 - copie.jpg" width="400" title="Input"> | <img src="static/image/outputs/s5.jpg" width="400" title="Style"> | <img src="static/image/outputs/output.png" width="400" title="Output">
-->
