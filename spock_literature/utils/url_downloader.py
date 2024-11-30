import urllib
import requests
import re
from stqdm import stqdm
import os
import shutil
import time
from bs4 import BeautifulSoup as bs
from datetime import datetime
from random import uniform as rand
import json
import numpy as np




class XRxivQuery:
    def __init__(self, search_query, max_results, folder_name='docs', XRxiv_servers = [], search_by='all', sort_by='relevance'):
       self.search_query = search_query
       self.max_results = max_results
       self.folder_name = folder_name
       self.XRxiv_servers = XRxiv_servers
       self.search_by = search_by
       self.sort_by = sort_by
       self.all_pdf_info = []
       self.all_pdf_citation = []

    def call_API(self):
        search_query = self.search_query.strip().replace(" ", "+").split('+')#.replace(", ", "+").replace(",", "+")#.split('+')
        max_papers_in_server = distibute_max_papers(self.max_results, self.XRxiv_servers)
        if 'rxiv' in self.XRxiv_servers:
            '''
            Scraps the arXiv's html to get data from each entry in a search. Entries has the following formatting:
            <entry>\n    
            <id>http://arxiv.org/abs/2008.04584v2</id>\n    
            <updated>2021-05-11T12:00:24Z</updated>\n    
            <published>2020-08-11T08:47:06Z</published>\n    
            <title>Bayesian Selective Inference: Non-informative Priors</title>\n    
            <summary>  We discuss Bayesian inference for parameters selected using the data. First,\nwe provide a critical analysis of the existing positions in the literature\nregarding the correct Bayesian approach under selection. Second, we propose two\ntypes of non-informative priors for selection models. These priors may be\nemployed to produce a posterior distribution in the absence of prior\ninformation as well as to provide well-calibrated frequentist inference for the\nselected parameter. We test the proposed priors empirically in several\nscenarios.\n</summary>\n    
            <author>\n      <name>Daniel G. Rasines</name>\n    </author>\n    <author>\n      <name>G. Alastair Young</name>\n    </author>\n    
            <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">24 pages, 7 figures</arxiv:comment>\n    
            <link href="http://arxiv.org/abs/2008.04584v2" rel="alternate" type="text/html"/>\n    
            <link title="pdf" href="http://arxiv.org/pdf/2008.04584v2" rel="related" type="application/pdf"/>\n    
            <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="math.ST" scheme="http://arxiv.org/schemas/atom"/>\n    
            <category term="math.ST" scheme="http://arxiv.org/schemas/atom"/>\n    
            <category term="stat.TH" scheme="http://arxiv.org/schemas/atom"/>\n  
            </entry>\n  
            '''
            # Call arXiv API
            journal = 'arXiv'
            max_rxiv_papers = max_papers_in_server[0]
            arXiv_url=f'http://export.arxiv.org/api/query?search_query={self.search_by}:{"+".join(search_query)}&sortBy={self.sort_by}&start=0&max_results={max_rxiv_papers}'
            with urllib.request.urlopen(arXiv_url) as url:
                s = url.read()
            
            # Parse the xml data
            from lxml import html
            root = html.fromstring(s)
            # Fetch relevant pdf information
            pdf_entries = root.xpath("entry")
            pdf_titles   = []
            pdf_authors  = []
            pdf_urls     = []
            pdf_categories = []
            folder_names = []
            pdf_citation = []
            pdf_years = []
            for i, pdf in enumerate(pdf_entries):
                pdf_titles.append(re.sub('[^a-zA-Z0-9]', ' ', pdf.xpath("title/text()")[0]))
                pdf_authors.append(pdf.xpath("author/name/text()"))
                pdf_urls.append(pdf.xpath("link[@title='pdf']/@href")[0])
                pdf_categories.append(pdf.xpath("category/@term"))
                folder_names.append(self.folder_name)
                pdf_years.append(pdf.xpath('updated/text()')[0][:4])
                pdf_citation.append(f"{', '.join(pdf_authors[i])}, {pdf_titles[i]}. {journal} [{pdf_categories[i][0]}] ({pdf_years[i]}), (available at {pdf_urls[i]}).")
            pdf_info = list(zip(pdf_titles, pdf_urls, pdf_authors, pdf_categories, folder_names, pdf_citation))
            self.all_pdf_info.append(pdf_info)

        if 'chemrxiv' in self.XRxiv_servers:
            '''
            See https://chemrxiv.org/engage/chemrxiv/public-api/documentation#tag/public-apiv1items/operation/getPublicapiV1Items
            '''
            # Call chemrxiv API
            journal = 'chemRxiv'
            max_chemrxiv_papers = max_papers_in_server[1]
            chemrxiv_url = f'https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term="{"%20".join(search_query)}"&sort=RELEVANT_DESC&limit={max_chemrxiv_papers}'
            req = urllib.request.Request(
                    url=chemrxiv_url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
            s = urllib.request.urlopen(req).read()
            jsonResponse = json.loads(s.decode('utf-8'))
            pdf_titles   = []
            pdf_authors  = []
            pdf_urls     = []
            pdf_categories = []
            folder_names = []
            pdf_citation = []
            pdf_years = []
            for i,d in enumerate(jsonResponse['itemHits']):
                pdf_titles.append(d['item']['title'].replace("\n", ""))
                authors_dict = d['item']['authors'] 
                pdf_authors.append([n['firstName']+' '+ n['lastName'] for n in authors_dict])
                pdf_urls.append('https://chemrxiv.org/engage/chemrxiv/article-details/'+ str(d['item']['id']))
                pdf_categories.append(journal)
                folder_names.append(self.folder_name)
                pdf_years.append(d['item']['statusDate'][:4])
                pdf_citation.append(f"{', '.join(pdf_authors[i])}, {pdf_titles[i]}. {journal} [{pdf_categories[i][0]}] ({pdf_years[i]}), (available at {pdf_urls[i]}).")
                # overwriting url cause chermRxiv sucks!
                pdf_urls[i] = d['item']['asset']['original']['url']
            pdf_info = list(zip(pdf_titles, pdf_urls, pdf_authors, pdf_categories, folder_names, pdf_citation))
            self.all_pdf_info.append(pdf_info)   


        if 'biorxiv' in self.XRxiv_servers or 'medrxiv' in self.XRxiv_servers:
            '''
            Scraps the biorxiv and medrxiv's html to get data from each entry in a search. Entries has the following formatting:
            <li class="first last odd search-result result-jcode-medrxiv search-result-highwire-citation">
            <div class="highwire-article-citation highwire-citation-type-highwire-article node" data-apath="/medrxiv/early/2021/02/18/2021.02.12.21251663.atom" data-pisa="medrxiv;2021.02.12.21251663v1" data-pisa-master="medrxiv;2021.02.12.21251663" id="node-medrxivearly202102182021021221251663atom1512875027"><div class="highwire-cite highwire-cite-highwire-article highwire-citation-biorxiv-article-pap-list clearfix">
            <span class="highwire-cite-title">
            <a class="highwire-cite-linked-title" data-hide-link-title="0" data-icon-position="" href="http://medrxiv.org/content/early/2021/02/18/2021.02.12.21251663">
            <span class="highwire-cite-title">ClinGen Variant Curation Interface: A Variant Classification Platform for the Application of Evidence Criteria from ACMG/AMP Guidelines</span></a> </span>
            <div class="highwire-cite-authors"><span class="highwire-citation-authors">
            <span class="highwire-citation-author first" data-delta="0"><span class="nlm-given-names">Christine G.</span> <span class="nlm-surname">Preston</span></span>,
            <span class="highwire-citation-author" data-delta="1"><span class="nlm-given-names">Matt W.</span> <span class="nlm-surname">Wright</span></span>, 
            <span class="highwire-citation-author" data-delta="2"><span class="nlm-given-names">Rao</span> <span class="nlm-surname">Madhavrao</span></span>, 
            <div class="highwire-cite-metadata"><span class="highwire-cite-metadata-journal highwire-cite-metadata">medRxiv </span>
            <span class="highwire-cite-metadata-pages highwire-cite-metadata">2021.02.12.21251663; </span><span class="highwire-cite-metadata-doi highwire-cite-metadata">
            <span class="doi_label">doi:</span> https://doi.org/10.1101/2021.02.12.21251663 </span></div>
            <div class="highwire-cite-extras"><div class="hw-make-citation" data-encoded-apath=";medrxiv;early;2021;02;18;2021.02.12.21251663.atom" data-seqnum="0" id="hw-make-citation-0">
            <a class="link-save-citation-save use-ajax hw-link-save-unsave-catation link-icon" href="/highwire-save-citation/saveapath/%3Bmedrxiv%3Bearly%3B2021%3B02%3B18%3B2021.02.12.21251663.atom/nojs/0" id="link-save-citation-toggle-0" title="Save">
            <span class="icon-plus"></span> <span class="title">Add to Selected Citations</span></a></div></div>
            </div>
            </div></li> 
            </entry>\n  
            '''
            if 'biorxiv' in self.XRxiv_servers and 'medrxiv' not in self.XRxiv_servers:
                # print('Searching biorxiv\n')
                max_biorxiv_papers = max_papers_in_server[2]
                journals_str = f'%20jcode%3Abiorxiv'
            if 'biorxiv' not in self.XRxiv_servers and 'medrxiv' in self.XRxiv_servers:
                # print('Searching medrxiv\n')
                max_biorxiv_papers = max_papers_in_server[3]
                journals_str = f'%20jcode%3Amedrxiv'
            if 'biorxiv' in self.XRxiv_servers and 'medrxiv' in self.XRxiv_servers:
                # print('Searching both biorxiv and medrxiv\n')
                max_biorxiv_papers = max_papers_in_server[3]+ max_papers_in_server[2] # birxiv and medrxiv are together.
                journals_str = f'%20jcode%3Abiorxiv%7C%7Cmedrxiv'
            
            subject_str = ('%20').join(self.search_query[0].split())
            for subject in search_query[1:]:
                subject_str = subject_str + '%252B' + ('%20').join(subject.split())
            
            current_dateTime = datetime.now()
            today = str(current_dateTime)[:10]
            start_day = '2013-01-01'
            arXiv_url = f'https://www.biorxiv.org/search/'
            arXiv_url += subject_str + journals_str + f'%20limit_from%3A2{start_day}%20limit_to%3A{today}%20numresults%3A{max_biorxiv_papers}%20sort%3Arelevance-rank%20format_result%3Astandard'

            url_response = requests.post(arXiv_url)
            html = bs(url_response.text, features='html.parser')
            pdf_entries = html.find_all(attrs={'class': 'search-result'})
            pdf_titles   = []
            pdf_authors  = []
            pdf_urls     = []
            pdf_categories = []
            folder_names = []
            pdf_citation = []
            pdf_years = []
            for i, pdf in enumerate(pdf_entries):
                pdf_titles.append(pdf.find('span', attrs={'class': 'highwire-cite-title'}).text.strip())
                pdf_authors.append(pdf.find('span', attrs={'class': 'highwire-citation-authors'}).text.strip().split(', '))
                pdf_url = pdf.find('a', href=True)['href']
                if pdf_url[:4] != 'http':
                    pdf_url = f'http://www.biorxiv.org'+ pdf_url
                pdf_urls.append(pdf_url)
                pdf_categories.append(pdf.find('span', attrs={'class': 'highwire-cite-metadata-journal highwire-cite-metadata'}).text.strip())
                folder_names.append(self.folder_name)
                pdf_years.append(pdf.find('span', attrs={'class': 'highwire-cite-metadata-pages highwire-cite-metadata'}).text.strip()[:4])
                pdf_citation.append(f"{', '.join(pdf_authors[i])}, {pdf_titles[i]}. {pdf_categories[i]} ({pdf_years[i]}), (available at {pdf_urls[i]}).")

            pdf_info = list(zip(pdf_titles, pdf_urls, pdf_authors, pdf_categories, folder_names, pdf_citation))
            self.all_pdf_info.append(pdf_info)

        self.all_pdf_info = [item for sublist in self.all_pdf_info for item in sublist]
        return self.all_pdf_info

    def download_pdf(self):
        all_reference_text = []
        for i,p in enumerate(stqdm(self.all_pdf_info, desc='🔍 Searching and downloading papers')):
            pdf_title=p[0]
            pdf_category=p[3]
            pdf_url=p[1]
            if pdf_category in ['medRxiv', 'bioRxiv']:
                pdf_url += '.full.pdf'
            pdf_file_name=p[0].replace(':','').replace('/','').replace('.','').replace('\n','')
            folder_name=p[4]
            pdf_citation=p[5]
            r = requests.get(pdf_url, allow_redirects=True)
            if  i == 0:
                if not os.path.exists(f'{folder_name}'):
                    os.makedirs(f"{folder_name}")
                else:
                    shutil.rmtree(f'{folder_name}') 
                    os.makedirs(f"{folder_name}")
            with open(f'{folder_name}/{pdf_file_name}.pdf', 'wb') as f:
                f.write(r.content)
            if i == 0:
                st.markdown("###### Papers found:")
            st.markdown(f"{i+1}. {pdf_citation}")
            time.sleep(0.15)
            all_reference_text.append(f"{i+1}. {pdf_citation}\n")
        if 'all_reference_text' not in st.session_state:
            st.session_state.key = 'all_reference_text'
        st.session_state['all_reference_text'] = ' '.join(all_reference_text)


                        
def distibute_max_papers(max_results, XRxiv_servers):
    fixed_length = len(XRxiv_servers)
    sample = np.random.multinomial(max_results - fixed_length, np.ones(fixed_length)/fixed_length, size=1)[0] + 1
    max_papers_in_server = np.zeros(4, dtype=int)
    all_servers = ['rxiv', 'chemrxiv', 'biorxiv', 'medrxiv']
    for i,s in enumerate(XRxiv_servers):
        max_papers_in_server[all_servers.index(s)] = int(sample[i])
    return  max_papers_in_server     
