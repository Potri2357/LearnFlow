import urllib.request
import os

urls = {
  "dashboard.html": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzc0NzY3NTAyNDAzNjRkZDBiZTNkYzIzZWQzNDdjMzgwEgsSBxCr-Z6M-wkYAZIBIwoKcHJvamVjdF9pZBIVQhM5MTI3ODM0NzI1NDM3MjkyMTEx&filename=&opi=89354086",
  "exam_prep.html": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzQ3MzlhMDU3YzJkNDRkMTc4ODY1Y2IzOGM0MDEzMWY5EgsSBxCr-Z6M-wkYAZIBIwoKcHJvamVjdF9pZBIVQhM5MTI3ODM0NzI1NDM3MjkyMTEx&filename=&opi=89354086",
  "profile.html": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzYzZDk4OGM2ZDVlYTRmZGE5NTMyMTRmMWNmZTAyNWRlEgsSBxCr-Z6M-wkYAZIBIwoKcHJvamVjdF9pZBIVQhM5MTI3ODM0NzI1NDM3MjkyMTEx&filename=&opi=89354086"
}

os.makedirs("stitch_html", exist_ok=True)
for name, url in urls.items():
  print(f"Downloading {name}...")
  urllib.request.urlretrieve(url, os.path.join("stitch_html", name))
print("Done.")
