import urllib.request
import urllib.error
import urllib.parse


headers = {
    'Accept': 'application/json, text/plain, */*',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63'
}
username = 'admin'
password = '2022milesight'
url = f'http://localhost/vb.htm'

passmgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
passmgr.add_password(None, url, username, password)
handler = urllib.request.HTTPDigestAuthHandler(passmgr)
opener = urllib.request.build_opener(handler)

for speed in range(1, 10):
  request_url = f'http://localhost/vb.htm?&ptzspeed={speed}'
  request = urllib.request.Request(request_url, headers=headers)

  try:
      with opener.open(request) as response:
          print(response.status)
          print(response.read().decode('utf-8'))
  except urllib.error.URLError as e:
      print(e)
  except:
      print('Unknown error')
