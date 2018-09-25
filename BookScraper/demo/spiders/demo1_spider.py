import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            #'http://quotes.toscrape.com/page/1/',
            #'http://quotes.toscrape.com/page/2/',
            'https://www.goodreads.com/genres/most_read/science-fiction/'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # page = response.url.split("/")[-2]
        # filename = 'quotes-%s.html' % page
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)
        for title in response.css('div.coverWrapper'):
            yield {
                'title': title.css('img::attr(alt)').extract_first()
            }