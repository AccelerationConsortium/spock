import abc

class PDF_analysis(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Complete analysis: Scan the paper and asks the questions we already have
        Chat with PDF: Chat with the PDF with custom user questions        
        """
        return (hasattr(subclass, 'complete_analysis') and # Questions we had
                callable(subclass.complete_analysis) and 
                hasattr(subclass, 'chat_with_pdf') and # Chat with the PDF with custom questions
                callable(subclass.chat_with_pdf))