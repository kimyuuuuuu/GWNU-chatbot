import requests
from bs4 import BeautifulSoup

def fetch_table_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # department-table에서 <tr> 태그 추출
            department_table = soup.find('table', class_='department-table')
            department_rows = department_table.find_all('tr') if department_table else []

            for row in department_rows:
                cells = row.find_all('td')
                for cell in cells:
                    # a 태그가 있을 경우 링크와 함께 출력
                    if cell.find('a'):
                        link = cell.find('a')['href']
                        text = cell.get_text(strip=True)
                        print(f"{text} ({link})")
                    else:
                        print(cell.get_text(strip=True))

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# 예시 URL (실제 URL로 변경 필요)
url = "https://www.gwnu.ac.kr/kr/7730/subview.do"
fetch_table_data(url)
